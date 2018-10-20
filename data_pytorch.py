import os
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset


def dummy_augment(index, image, mask):
    return index, image, mask


class TsgDataset(Dataset):
    def __init__(self, root, image_ids, augment, mode='train'):
        self.root = root
        self.image_ids = image_ids
        self.mode = mode
        self.augment = augment

        self.depths = pd.read_csv('../input/depths.csv', index_col='id')

    def __getitem__(self, index):
        mask = None
        image_id = self.image_ids[index]
        image = cv2.imread(os.path.join(self.root, 'images/{}.png'.format(image_id)),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

        if self.mode in ['train', 'valid']:
            mask = cv2.imread(os.path.join(self.root, 'masks/{}.png'.format(image_id)),
                              cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        elif self.mode in ['test']:
            mask = np.array([])

        label = 0 if np.sum(mask == 1) == 0 else 1
        depth = self.depths.loc[image_id]['z']

        index, image, mask = self.augment(index, image, mask)

        # 添加深度信息
        # image = self._add_depth_channels(image)
        # image = np.transpose(image, (2, 0, 1))

        return index, image, mask, label

    def __len__(self):
        return len(self.image_ids)

    def _add_depth_channels(self, image):
        h, w = image.shape
        image = np.stack([image, image, image], 2)
        for row, const in enumerate(np.linspace(0., 1, h)):
            image[row, :, 1] = const
        image[:, :, 2] = image[:, :, 0] * image[:, :, 1]
        return image
