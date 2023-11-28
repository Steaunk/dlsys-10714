from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        with gzip.open(image_filename, "rb") as img_file:
            magic_num, img_num, row, col = struct.unpack(
                ">4i", img_file.read(16))
            tot_pixels = row * col
            X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(
                tot_pixels)), dtype=np.float32) for _ in range(img_num)])
            X -= np.min(X)
            X /= np.max(X)

        with gzip.open(label_filename, "rb") as label_file:
            magic_num, label_num = struct.unpack(">2i", label_file.read(8))
            y = np.array(struct.unpack(
                f"{label_num}B", label_file.read()), dtype=np.uint8)

        self.images = X
        self.labels = y
        self.row = row
        self.col = col
        super().__init__(transforms)
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        img = self.images[index]
        label = self.labels[index]
        if len(img.shape) > 1:
            img = np.array([self.apply_transforms(
                i.reshape(self.row, self.col, 1)).reshape(i.shape) for i in img])
        else:
            img = self.apply_transforms(img.reshape(
                self.row, self.col, 1)).reshape(img.shape)
        return (img, label)
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return self.images.shape[0]
        # raise NotImplementedError()
        # END YOUR SOLUTION
