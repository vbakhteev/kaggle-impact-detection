import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from effdet import DetBenchPredict
from tqdm import tqdm

from config import data as data_cfg, model as model_cfg, train as train_cfg
from src.data.loaders import get_dataloaders
from src.metric import comp_metric
from src.model.detector import get_net
from src.utils import seed_everything, AverageMeter


def main():
    seed_everything()
    train()


def train():
    loader_train, loaders_valid_fn = get_dataloaders(data_cfg, train_cfg)
    mid_frame = np.where(np.array(data_cfg.frames_neighbors) == 0)[0][0]

    device = torch.device('cuda:0')
    net = get_net(model_cfg, num_classes=2, mid_frame=mid_frame).to(device)
    fitter = Fitter(model=net, device=device)

    try:
        fitter.fit(loader_train, loaders_valid_fn)
    except KeyboardInterrupt:
        pass

    fitter.store_best_parameters()


class Fitter:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        checkpoints_dir = model_cfg.efficientdet_config + datetime.now().strftime("_%B%d_%H:%M:%S")
        self.base_dir = Path('checkpoints') / checkpoints_dir
        os.makedirs(self.base_dir)
        self.log_path = self.base_dir / 'log.txt'

        self.epoch = 0
        self.best_metric = -1
        self.early_stopping_patience = train_cfg.early_stopping_patience
        self.epochs_without_improvement = 0

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-6},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_cfg.lr)
        self.scheduler = train_cfg.SchedulerClass(self.optimizer, **train_cfg.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loaders_fn):
        for e in range(train_cfg.n_epochs):
            if train_cfg.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            summary_loss, summary_class_loss, summary_box_loss = self.train_one_epoch(train_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, '
                     f'loss: {summary_loss.avg:.5f}, '
                     f'class_loss: {summary_class_loss.avg:.5f}, '
                     f'box_loss: {summary_box_loss.avg:.5f}')

            threshold, metric, videos_scores = self.validation(validation_loaders_fn)
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, threshold: {threshold}, metric: {metric:.5f}')
            self.save(self.base_dir / 'last-checkpoint.bin')
            if train_cfg.validation_scheduler:
                self.scheduler.step(metrics=metric)

            if metric > self.best_metric:
                self.best_metric = metric
                self.model.eval()
                self.save(self.base_dir / f'best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(self.base_dir.glob('best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement > self.early_stopping_patience:
                msg = f'Finishing training because val loss did not improved ' \
                      f'for last {self.epochs_without_improvement} epochs'
                self.log(msg)
                return

            self.epoch += 1

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss, summary_class_loss, summary_box_loss = [AverageMeter() for _ in range(3)]

        it = tqdm(train_loader, total=len(train_loader))
        for images, targets in it:
            msg = f'Train loss: {summary_loss.avg:.5f}, ' \
                  f'class_loss: {summary_class_loss.avg:.5f}, ' \
                  f'box_loss: {summary_box_loss.avg:.5f}'
            it.set_description(msg)

            loss, class_loss, box_loss, _ = self.one_forward(images, targets)

            # Faster than optimizer.zero_grad()
            for param in self.model.parameters():
                param.grad = None

            loss.backward()
            self.optimizer.step()
            if train_cfg.step_scheduler:
                self.scheduler.step()

            bs = len(images)
            summary_loss.update(loss.item(), bs)
            summary_class_loss.update(class_loss.item(), bs)
            summary_box_loss.update(box_loss.item(), bs)

        return summary_loss, summary_class_loss, summary_box_loss

    def one_forward(self, images, targets):
        images = torch.stack(images)
        images = images.to(self.device).float()

        boxes = [target['boxes'].to(self.device).float() for target in targets]
        labels = [target['labels'].to(self.device).float() for target in targets]
        target = dict(
            bbox=boxes,
            cls=labels,
            img_scale=None,
            img_size=None,
        )

        output = self.model(images, target)
        return output['loss'], output['class_loss'], output['box_loss'], output.get('detections', None)

    def validation(self, val_loaders_fn):
        self.model.eval()
        inference_model = DetBenchPredict(self.model.model).to(self.device)

        gt, preds, scores = [], [], []
        for val_loader in tqdm(val_loaders_fn(), total=24):
            gt_video, video_preds, video_scores = self.predict_single_video(
                inference_model=inference_model, val_loader=val_loader,
            )
            preds += [video_preds]
            scores += [video_scores]

        best_threshold, _, _ = self.find_best_threshold(
            thresholds=np.arange(0, 1, 0.05),
            gt=gt,
            preds=preds,
            scores=scores
        )
        best_threshold, best_metric, videos_best_scores = self.find_best_threshold(
            thresholds=np.arange(best_threshold - 0.04, best_threshold + 0.04, 0.01),
            gt=gt,
            preds=preds,
            scores=scores
        )

        return best_threshold, best_metric, videos_best_scores

    def predict_single_video(self, inference_model, val_loader):
        video_gt_boxes, video_boxes, video_scores, video_frames = [], [], [], []

        for images, gt_boxes, frame_ids in val_loader:
            images = torch.stack(images)
            images = images.to(self.device).float()

            with torch.no_grad():
                output = inference_model(images)
                output = output.cpu().numpy()

            height_scale = 720 / model_cfg.img_size[0]
            width_scale = 1280 / model_cfg.img_size[1]
            output[:, :, 0] = output[:, :, 0] * width_scale
            output[:, :, 2] = output[:, :, 2] * width_scale
            output[:, :, 1] = output[:, :, 1] * height_scale
            output[:, :, 3] = output[:, :, 3] * height_scale

            # Keep only accident boxes and corresponding scores
            labels = output[:, :, 5]
            cond = (labels != 1)
            indexes = [np.where(c)[0] for c in cond]
            output = [
                sample_output[sample_indexes]
                for sample_output, sample_indexes in zip(output, indexes)
            ]
            boxes = [o[:, :4].astype(int) for o in output]
            scores = [o[:, 4] for o in output]
            video_boxes += boxes
            video_scores += scores

            # Keep GT boxes and frame_ids
            for gt_boxes_sample, frame_id in zip(gt_boxes, frame_ids):
                if gt_boxes_sample.shape[0] > 0:
                    video_gt_boxes.append(gt_boxes_sample)
                    video_frames += [frame_id] * gt_boxes_sample.shape[0]

        # Concatenate frame's ids and corresponding GT boxes
        video_gt_boxes = np.concatenate(video_gt_boxes, axis=0)
        video_frames = np.array(video_frames)[:, np.newaxis]
        gt_video = np.concatenate((video_frames, video_gt_boxes), axis=1)

        # Concatenate frame's ids and corresponding predicted boxes
        video_preds = []
        for frame_id, boxes_sample in enumerate(video_boxes):
            frame_id_repeated = np.full((boxes_sample.shape[0], 1), frame_id)
            frame_and_boxes = np.concatenate((frame_id_repeated, boxes_sample), axis=1)
            video_preds += [frame_and_boxes]

        # Concat predictions of 1 video into array
        video_preds = np.concatenate(video_preds, axis=0)  # (N_predicted_boxes, 5)
        video_scores = np.concatenate(video_scores, axis=0)  # (N_predicted_boxes,)

        return gt_video, video_preds, video_scores

    def find_best_threshold(self, thresholds, gt, preds, scores):
        best_metric = best_threshold = -1
        videos_best_scores = None

        for threshold in thresholds:

            thresholded_preds = []
            for video_preds, video_scores in zip(preds, scores):
                thresholded_video_preds = video_preds[video_scores >= threshold]
                thresholded_preds += [thresholded_video_preds]

            precision, recall, f1_score, f1_per_video = comp_metric(thresholded_preds, gt)
            if f1_score > best_metric:
                best_metric = f1_score
                best_threshold = threshold
                videos_best_scores = f1_per_video

        return best_threshold, best_metric, videos_best_scores

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_metric = checkpoint['best_metric']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if train_cfg.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

    def store_best_parameters(self):
        best_weights_path = sorted(self.base_dir.glob('best-checkpoint-*epoch.bin'))[-1]
        self.load(best_weights_path)
        torch.save(
            {'state_dict': self.model.model.state_dict()},
            self.base_dir / 'best_weights.bin'
        )


if __name__ == '__main__':
    main()
