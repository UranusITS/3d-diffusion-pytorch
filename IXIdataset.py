import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from normalization import CTNormalization

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

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
    H = 150
    W = 256
    def __init__(self, data_dir='/data_hdd/users/lisikuang/IXI/T1/train/', depth=256):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        self.depth = depth
        self.normalization = CTNormalization(
            False,
            {
                'mean': 262.046,
                'std': 616.704,
                'percentile_00_5': 50,
                'percentile_99_5': 6000,
            }
        )
        self.pad_width = [(0, 0), ((self.W - self.H) // 2, (self.W - self.H) // 2)] # padding (256, 150) -> (256, 256)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.W // 2, self.W // 2), antialias=True)
        ])

    def __len__(self):
        return len(self.data_files) * (self.depth - 2)

    def image_process(self, image: np.ndarray) -> torch.Tensor:
        image = np.pad(image, self.pad_width)
        # image = np.resize(image, (self.W // 2, self.W // 2))
        image = self.normalization.run(image)
        image = self.to_tensor(image)
        return image

    def __getitem__(self, idx):
        data_idx = idx // (self.depth - 2)
        depth_idx = idx % (self.depth - 2)

        depths = torch.tensor((depth_idx, depth_idx + 1), dtype=torch.float32)

        data_path = os.path.join(self.data_dir, self.data_files[data_idx])
        data_obj = nib.load(data_path)
        data_array = data_obj.get_fdata()
        data_array = data_array.transpose(1, 0, 2)

        lower_image = self.image_process(data_array[depth_idx])
        label_image = self.image_process(data_array[depth_idx + 1])
        upper_image = self.image_process(data_array[depth_idx + 2])

        images = torch.stack((lower_image, upper_image, label_image), dim=0)

        return images, depths


if __name__ == '__main__':
    dataset = IXIDataset()
    datas = dataset[0]

    for data in datas:
        print(data.shape)
        print(data)
