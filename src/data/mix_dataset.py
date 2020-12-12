import random

from torch.utils.data import Dataset

from src.data.transforms import mixup_video, cutmix_video, cutmix_image


class MixDataset(Dataset):
    def __init__(self, dataset, video=True):
        self.dataset = dataset
        self.cutmix_fn = cutmix_video if video else cutmix_image

    def __getitem__(self, index):
        images1, target1 = self.dataset[index]
        boxes1, labels1 = target1['boxes'], target1['labels']

        random_index = random.randint(0, len(self.dataset) - 1)
        images2, target2 = self.dataset[random_index]
        boxes2, labels2 = target2['boxes'], target2['labels']

        if random.random() < 0.5:
            images, boxes, labels = mixup_video(
                images1, images2, boxes1, boxes2, labels1, labels2
            )
        else:
            images, boxes, labels = self.cutmix_fn(
                images1, images2, boxes1, boxes2, labels1, labels2
            )

        return images, {'boxes': boxes, 'labels': labels}

    def __len__(self):
        return len(self.dataset)
