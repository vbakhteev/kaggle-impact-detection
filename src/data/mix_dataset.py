import random

from torch.utils.data import Dataset

from src.data.transforms import mixup, cutmix_video, cutmix_image


class MixDataset(Dataset):
    def __init__(self, dataset, video=True):
        self.dataset = dataset
        self.video = video

    def __getitem__(self, index):
        images1, target1 = self.dataset[index]
        boxes1, labels1 = target1['boxes'], target1['labels']

        if random.random() < 0.333:
            return images1, target1

        random_index = random.randint(0, len(self.dataset) - 1)
        images2, target2 = self.dataset[random_index]
        boxes2, labels2 = target2['boxes'], target2['labels']

        # MixUp
        if random.random() < 0.5:
            images, boxes, labels = mixup(
                images1, images2, boxes1, boxes2, labels1, labels2
            )

        # CutMix
        else:
            samples = [
                (images1, boxes1, labels1),
                (images2, boxes2, labels2),
            ]

            if self.video:
                images, boxes, labels = cutmix_video(samples)

            # If image then construct mosaic of 4 images
            else:
                for _ in range(2):
                    random_index = random.randint(0, len(self.dataset) - 1)
                    images_i, target_i = self.dataset[random_index]
                    boxes_i, labels_i = target_i['boxes'], target_i['labels']
                    samples += [(images_i, boxes_i, labels_i)]
                images, boxes, labels = cutmix_image(samples)

        return images, {'boxes': boxes, 'labels': labels}

    def __len__(self):
        return len(self.dataset)
