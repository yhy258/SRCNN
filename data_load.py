from torch.utils.data import Dataset
from skimage import io
import os
import torch.nn as nn


class SR_dataset(Dataset):
    def __init__(self, lr_path, hr_path, transform, interpolation_scale=4, interpolation_mode='bicubic'):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.transform = transform
        self.lr_image_names = os.listdir(lr_path)
        self.hr_image_names = os.listdir(hr_path)
        self.interpolation_scale = interpolation_scale
        self.interpolation_mode = interpolation_mode
        self.upsample = nn.Upsample(scale_factor=interpolation_scale, mode='bicubic')

    def __len__(self):
        return len(self.lr_image_names)

    def __getitem__(self, i):
        image_name = self.lr_image_names[i]
        lr_path = self.lr_path + "/" + image_name
        hr_path = self.hr_path + "/" + image_name
        lr_image = io.imread(lr_path)
        hr_image = io.imread(hr_path)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        # low resolution은 그냥 upsampling 하고 보내자.

        lr_image_size = lr_image.size()
        lr_image = self.upsample(lr_image.view(1, *lr_image_size))

        height = lr_image_size[1]
        width = lr_image_size[2]
        height *= self.interpolation_scale
        width *= self.interpolation_scale

        lr_image = lr_image.view(3, height, width)

        return lr_image, hr_image