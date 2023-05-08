import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from normalization import CTNormalization


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class IXIDataset(Dataset):
    def __init__(self,
                 data_dir='/data_hdd/users/lisikuang/IXI/T1/train/',
                 normalazation=CTNormalization(
                     False,
                     {
                         'mean': 262.046,
                         'std': 616.704,
                         'percentile_00_5': 50,
                         'percentile_99_5': 6000,
                     }
                 ),
                 depth=256,
                 input_width=128,
                 output_width=256):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        self.depth = depth
        self.input_width = input_width
        self.output_width = output_width
        self.normalization = normalazation
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
        return len(self.data_files) * (self.depth - 2)

    def image_process(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape
        pad_width = [(0, 0), ((H - W) // 2, (H - W) - (H - W) // 2)
                     ] if H > W else [((W - H) // 2, (W - H) - (W - H) // 2), (0, 0)]
        image = np.pad(image, pad_width)
        image = self.normalization.run(image)
        return image

    def __getitem__(self, idx):
        data_idx = idx // (self.depth - 2)
        depth_idx = idx % (self.depth - 2)

        data_path = os.path.join(self.data_dir, self.data_files[data_idx])
        data_obj = nib.load(data_path)

        T = torch.tensor((data_obj.header.get_zooms()[
                         1] * 2), dtype=torch.float32)

        data_array = data_obj.get_fdata()
        data_array = data_array.transpose(1, 0, 2)

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
    dataset = IXIDataset()
    datas = dataset[0]

    for data in datas:
        print(data.shape)
        # print(data)
