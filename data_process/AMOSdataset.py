import json
import nibabel as nib
import numpy as np
import os
import torch
from bisect import bisect
from .normalization import ImageNormalization, CTNormalization
from torch.utils.data import Dataset
from torchvision import transforms


class AMOSDataset(Dataset):
    def __init__(
        self,
        mode='training',
        data_dir='/data_hdd/users/lisikuang/amos22/',
        normalization: ImageNormalization = CTNormalization(
            False,
            {
                'mean': -277.6655390597008,
                'std': 7598.694741902908,
                'percentile_00_5': -3024,
                'percentile_99_5': 23872,
            }
        ),
        input_width=128,
        output_width=256
    ):
        self.mode = mode
        self.data_dir = data_dir
        self.json_file = os.path.join(data_dir, 'dataset.json')
        self.image_files = []
        # self.label_files = []
        self.depth_sums = []
        self.length = 0
        with open(self.json_file) as json_file:
            json_data = json.load(json_file)
            for image_label_pair in json_data[mode]:
                image_file = os.path.join(data_dir, image_label_pair['image'])
                # label_file = os.path.join(data_dir, image_label_pair['label'])
                image_obj = nib.load(image_file)
                self.length += image_obj.header['dim'][3] - 2
                self.image_files.append(image_file)
                # self.label_files.append(label_file)
                self.depth_sums.append(self.length)

        self.input_width = input_width
        self.output_width = output_width
        self.normalization = normalization

        self.input_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.input_width, self.input_width), antialias=True)
        ])

        self.output_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.output_width, self.output_width), antialias=True)
        ])

    def __len__(self):
        return self.length

    def image_process(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape
        pad_width = [(0, 0), ((H - W) // 2, (H - W) - (H - W) // 2)] if H > W \
                    else [((W - H) // 2, (W - H) - (W - H) // 2), (0, 0)]
        image = np.pad(image, pad_width)
        image = self.normalization.run(image)
        return image

    def __getitem__(self, idx):
        data_idx = bisect(self.depth_sums, idx)
        depth_idx = idx if data_idx == 0 \
                    else idx - self.depth_sums[data_idx - 1]

        data_path = os.path.join(self.data_dir, self.image_files[data_idx])
        data_obj = nib.load(data_path)

        T = torch.tensor(
            (data_obj.header.get_zooms()[1] * 2),
            dtype=torch.float32
        )

        data_array = data_obj.get_fdata()
        data_array = data_array.transpose(2, 0, 1)

        lower_image = self.image_process(data_array[depth_idx])
        lower_image = self.input_transforms(lower_image)
        label_image = self.image_process(data_array[depth_idx + 1])
        label_image = self.output_transforms(label_image)
        upper_image = self.image_process(data_array[depth_idx + 2])
        upper_image = self.input_transforms(upper_image)

        X = torch.stack((lower_image, upper_image), dim=0)
        Y = label_image

        return X, Y, T


if __name__ == '__main__':
    dataset = AMOSDataset()
    datas = dataset[0]

    for data in datas:
        print(data.shape)
        # print(data)
