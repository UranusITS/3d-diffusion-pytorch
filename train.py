import argparse
from xunet import XUNet

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from utils import *

from tqdm import tqdm
from einops import rearrange
import time
from math import isinf

from data_process.AMOSdataset import AMOSDataset
from data_process.IXIdataset import IXIDataset
from data_process.sampler import MultiEpochsDataLoader
from data_process.normalization import CTNormalization, RescaleTo01WithMinMaxNormalization
from tensorboardX import SummaryWriter
import os

input_width = 128
output_width = 256
batch_size = 4
test_batch_size = 32
lr = 1e-6
dataset_name = 'AMOS'

if dataset_name == 'IXI':
    normalization = CTNormalization(
        False,
        {
            'mean': 262.046,
            'std': 616.704,
            'percentile_00_5': 50,
            'percentile_99_5': 6000,
        }
    )
    d = IXIDataset('/data_hdd/users/lisikuang/IXI/T1/train/', normalization, input_width=input_width, output_width=output_width)
    d_val = IXIDataset('/data_hdd/users/lisikuang/IXI/T1/valid/', normalization, input_width=input_width, output_width=output_width)
elif dataset_name == 'AMOS':
    normalization = CTNormalization(
        False,
        {
            'mean': -277.6655390597008,
            'std': 7598.694741902908,
            'percentile_00_5': -3024,
            'percentile_99_5': 23872,
        }
    )
    d = AMOSDataset('training', normalization=normalization, input_width=input_width, output_width=output_width)
    d_val = AMOSDataset('validation', normalization=normalization, input_width=input_width, output_width=output_width)
else:
    raise NotImplementedError

loader = MultiEpochsDataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=40)
loader_val = DataLoader(d_val, batch_size=test_batch_size, shuffle=True, drop_last=True, num_workers=16)

device = 'cuda:0'

model = XUNet(H=output_width, W=output_width, input_ch=1, output_ch=1)
model = torch.nn.DataParallel(model)
model.to(device)

optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

def warmup(optimizer, step, last_step, last_lr):

    if step < last_step:
        optimizer.param_groups[0]['lr'] = step / last_step * last_lr

    else:
        optimizer.param_groups[0]['lr'] = last_lr


parser = argparse.ArgumentParser()
parser.add_argument('--transfer', type=str, default="")
args = parser.parse_args()


if args.transfer == "":
    tag = f'./results/{dataset_name}/int-hint-test'
    writer = SummaryWriter(tag)
    step = 0
else:
    print('transfering from: ', args.transfer)

    ckpt = torch.load(os.path.join(args.transfer, 'latest.pt'))

    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])

    tag = args.transfer
    writer = SummaryWriter(tag)
    step = ckpt['step']


for e in range(100000):
    print(f'starting epoch {e}')

    lt = time.time()
    for X, Y, T in tqdm(loader):

        # warmup(optimizer, step, 10000/batch_size, 0.0001)

        B = X.shape[0]

        optimizer.zero_grad()

        logsnr = logsnr_schedule_cosine(torch.rand((B,)))

        loss = p_losses(model, X=X.cuda(), Y=Y.cuda(), T=T, logsnr=logsnr.cuda(), loss_type="l2", cond_prob=0.1)
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), global_step=step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)

        if step % 500 == 0:
            print("Loss:", loss.item())

        if step % 1000 == 900:
        # if True: # test eval part
            model.eval()
            psnr_avg = 0
            psnr_cnt = 0
            ssim_avg = 0
            ssim_cnt = 0
            with torch.no_grad():
                for X, Y, T in loader_val:

                    b = test_batch_size // 8
                    w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(b)
                    img = sample(model, X, Y, T, w)

                    img = normalization.rerun(img[-1])
                    gt = normalization.rerun(Y)
                    cd = normalization.rerun(get_hint(X, (3, Y.shape[2], Y.shape[3])))
                    mn = float(min(img.min(), gt.min(), cd.min()))
                    mx = float(max(img.max(), gt.max(), cd.max()))
                    img = norm2255(img, mn, mx)
                    gt = norm2255(gt, mn, mx)
                    cd = norm2255(cd, mn, mx)

                    for i in range(test_batch_size):
                        psnr = compare_psnr(gt[i, 0], img[i, 0])
                        ssim = compare_ssim(gt[i, 0], img[i, 0], 11)
                        if not isinf(psnr):
                            psnr_avg += psnr
                            psnr_cnt += 1
                        ssim_avg += ssim
                        ssim_cnt += 1

                    fi = np.concatenate([cd, gt, img], axis=2)
                    for i, ww in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
                        writer.add_image(f"train/{ww}", fi[i], step)
                    break
            writer.add_scalar("valid/psnr", psnr_avg / psnr_cnt, global_step=step)
            writer.add_scalar("valid/ssim", ssim_avg / ssim_cnt, global_step=step)
            print('image sampled!')
            writer.flush()
            model.train()

        # if step == int(100000/batch_size):
        #     torch.save({'optim': optimizer.state_dict(
        #     ), 'model': model.state_dict(), 'step': step}, tag+f"/after_warmup.pt")

        step += 1
        starttime = time.time()

    torch.save({'optim': optimizer.state_dict(), 'model': model.state_dict(
    ), 'step': step, 'epoch': e}, tag+f"/latest.pt")
