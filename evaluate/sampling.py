import sys
sys.path.append("..")

import argparse
from xunet import XUNet

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
import time
from PIL import Image

from data_process.normalization import CTNormalization
from data_process.IXIdataset import IXIDataset, MultiEpochsDataLoader
from utils import *
from tensorboardX import SummaryWriter
import os

image_size = 128
batch_size = 4
test_batch_size = 32

normalization = CTNormalization(
    False,
    {
        'mean': 262.046,
        'std': 616.704,
        'percentile_00_5': 50,
        'percentile_99_5': 6000,
    }
)

d_val = IXIDataset('/data_hdd/users/lisikuang/IXI/T1/valid/', normalization, input_width=image_size, output_width=image_size)

loader_val = DataLoader(d_val, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=16)

device = 'cuda:0'

model = XUNet(H=image_size, W=image_size, input_ch=2, output_ch=2)
model = torch.nn.DataParallel(model)
model.to(device)

tag = './results/IXI/int-hint-01'

ckpt = torch.load(tag + '/latest.pt')
model.load_state_dict(ckpt['model'])
model.eval()

with torch.no_grad():
    for idx, (oriimg, depth) in enumerate(loader_val):

        b = test_batch_size // 8
        w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(b)
        img = sample(model, img=oriimg, depth=depth)

        img = rearrange(((img[-1].clip(-1, 1) + 1) * 127.5).astype(
            np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=b)

        x = torch.cat((oriimg[:, 0], oriimg[:, 1]), 1)
        z = torch.cat((oriimg[:, 2], oriimg[:, 2]), 1)
        gt = rearrange(((z + 1) * 127.5).detach().cpu().numpy().astype(
            np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=b)
        cd = rearrange(((x + 1) * 127.5).detach().cpu().numpy().astype(
            np.uint8), "(b a) c h w -> a c h (b w)", a=8, b=b)

        fi = np.concatenate([cd, gt, img], axis=2)
        print(fi.shape)
        
        os.makedirs(f'sampling/{idx}', exist_ok=True)
        Image.fromarray(((fi.transpose(1,2,0)+1)*127.5).astype(np.uint8)).save(f'sampling/{idx}/result.png')
        print(f'image{idx} sampled!')
