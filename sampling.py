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

from IXIdataset import IXIDataset, MultiEpochsDataLoader
from tensorboardX import SummaryWriter
import os

image_size = 128
batch_size = 4
test_batch_size = 32

d_val = IXIDataset('/data_hdd/users/lisikuang/IXI/T1/valid/')

loader_val = DataLoader(d_val, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=16)

device = "cuda:0"

model = XUNet(H=image_size, W=image_size, input_ch=2, output_ch=2)
model = torch.nn.DataParallel(model)
model.to(device)

ckpt = torch.load('./results/IXI/1682571061/latest.pt')
model.load_state_dict(ckpt['model'])
model.eval()

def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b

    return -2. * torch.log(torch.tan(a * t + b))


def xt2batch(x, logsnr, z, depth):
    b = x.shape[0]

    return {
        'x': x.cuda(),
        'z': z.cuda(),
        'logsnr': torch.stack([logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1).cuda(),
        'depth': depth.cuda(),
    }


def q_sample(z, logsnr, noise):

    # lambdas = logsnr_schedule_cosine(t)

    alpha = logsnr.sigmoid().sqrt()
    sigma = (-logsnr).sigmoid().sqrt()

    alpha = alpha[:, None, None, None]
    sigma = sigma[:, None, None, None]

    return alpha * z + sigma * noise


def p_losses(denoise_model, img, depth, logsnr, noise=None, loss_type="l2", cond_prob=0.1):

    B = img.shape[0]
    x = torch.cat((img[:, 0], img[:, 1]), 1)
    z = torch.cat((img[:, 2], img[:, 2]), 1)
    if noise is None:
        noise = torch.randn_like(x)

    z_noisy = q_sample(z=z, logsnr=logsnr, noise=noise)

    cond_mask = (torch.rand((B,)) > cond_prob).cuda()

    x_condition = torch.where(
        cond_mask[:, None, None, None], x, torch.randn_like(x))

    batch = xt2batch(x=x_condition, logsnr=logsnr, z=z_noisy, depth=depth)

    predicted_noise = denoise_model(batch, cond_mask=cond_mask.cuda())

    if loss_type == 'l1':
        loss = F.l1_loss(noise.to(device), predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise.to(device), predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise.to(device), predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def sample(model, img, depth, timesteps=256):
    x = torch.cat((img[:, 0], img[:, 1]), 1)
    img = torch.randn_like(x)
    imgs = []

    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(
        torch.linspace(1., 0., timesteps+1)[1:])

    # [1, ..., 0] = size is 257
    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts)):
        img = p_sample(model, x=x, z=img, depth=depth,
                       logsnr=logsnr, logsnr_next=logsnr_next, w=w)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def p_sample(model, x, z, depth, logsnr, logsnr_next, w):

    model_mean, model_variance = p_mean_variance(
        model, x=x, z=z, depth=depth, logsnr=logsnr, logsnr_next=logsnr_next, w=w)

    if logsnr_next == 0:
        return model_mean

    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def p_mean_variance(model, x, z, depth, logsnr, logsnr_next, w=2.0):

    strt = time.time()
    b = x.shape[0]
    w = w[:, None, None, None]

    c = - torch.special.expm1(logsnr - logsnr_next)

    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (
        -logsnr).sigmoid(), (-logsnr_next).sigmoid()

    alpha, sigma, alpha_next = map(
        lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))

    # batch = xt2batch(x, logsnr.repeat(b), z, R)
    batch = xt2batch(x, logsnr.repeat(b), z, depth)

    strt = time.time()
    pred_noise = model(batch, cond_mask=torch.tensor([True]*b)).detach().cpu()
    batch['x'] = torch.randn_like(x).cuda()
    pred_noise_unconditioned = model(
        batch, cond_mask=torch.tensor([False]*b)).detach().cpu()

    pred_noise_final = (1+w) * pred_noise - w * pred_noise_unconditioned

    z = z.detach().cpu()

    z_start = (z - sigma * pred_noise_final) / alpha
    z_start.clamp_(-1., 1.)

    model_mean = alpha_next * (z * (1 - c) / alpha + c * z_start)

    posterior_variance = squared_sigma_next * c

    return model_mean, posterior_variance


def warmup(optimizer, step, last_step, last_lr):

    if step < last_step:
        optimizer.param_groups[0]['lr'] = step / last_step * last_lr

    else:
        optimizer.param_groups[0]['lr'] = last_lr


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
