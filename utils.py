import cv2
from einops import rearrange
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


def norm201(img, mn, mx):
    return (img - mn) / max((mx - mn), 1)


def norm2255(img, mn, mx):
    if torch.is_tensor(img):
        if img.get_device() < 0:
            return (norm201(img, mn, mx) * 255).numpy().astype(np.uint8)
        else:
            return (norm201(img, mn, mx) * 255).detach().cpu().numpy().astype(np.uint8)
    else:
        return (norm201(img, mn, mx) * 255).numpy().astype(np.uint8)


def compare_ssim(gt, img, win_size, channel_axis=None):
    return structural_similarity(gt, img, win_size=win_size, channel_axis=channel_axis)


def compare_psnr(gt, img):
    mx = max(np.max(gt), np.max(img))
    mn = min(np.min(gt), np.min(img))
    return peak_signal_noise_ratio(norm201(gt, mn, mx), norm201(img, mn, mx))


def get_hint(img, size):
    hint = rearrange(img, 'n d c h w -> n c d h w')
    hint = F.interpolate(hint, size, mode='trilinear')
    mid = hint.shape[2] // 2
    hint = hint[:, :, mid, :, :]
    return hint


def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b
    return -2. * torch.log(torch.tan(a * t + b))


def xt2batch(x, z, logsnr, t):
    return {
        'x': x.cuda(),
        'z': z.cuda(),
        'logsnr': torch.stack([logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1).cuda(),
        't': t.cuda()
    }


def q_sample(z, logsnr, noise):
    alpha = logsnr.sigmoid().sqrt()
    sigma = (-logsnr).sigmoid().sqrt()
    alpha = alpha[:, None, None, None]
    sigma = sigma[:, None, None, None]
    return alpha * z + sigma * noise


def p_losses(denoise_model, X, Y, T, logsnr, noise=None, loss_type="l2", cond_prob=0.1, device='cuda:0'):
    B = X.shape[0]
    x = get_hint(X, (3, Y.shape[2], Y.shape[3]))
    z = Y
    if noise is None:
        noise = torch.randn_like(x)
    z_noisy = q_sample(z=z, logsnr=logsnr, noise=noise)
    cond_mask = (torch.rand((B,)) > cond_prob).cuda()
    x_condition = torch.where(
        cond_mask[:, None, None, None], x, torch.randn_like(x))
    batch = xt2batch(x=x_condition, z=z_noisy, logsnr=logsnr, t=T)
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
def sample(model, X, Y, T, w, timesteps=256):
    x = get_hint(X, (3, Y.shape[2], Y.shape[3]))
    img = torch.randn_like(x)
    imgs = []
    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(
        torch.linspace(1., 0., timesteps+1)[1:])
    # [1, ..., 0] = size is 257
    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts), total=len(logsnrs), leave=False):
        img = p_sample(model, x=x, z=img, t=T,
                       logsnr=logsnr, logsnr_next=logsnr_next, w=w)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def p_sample(model, x, z, t, logsnr, logsnr_next, w):
    model_mean, model_variance = p_mean_variance(
        model, x=x, z=z, t=t, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
    if logsnr_next == 0:
        return model_mean
    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def p_mean_variance(model, x, z, t, logsnr, logsnr_next, w=2.0):
    strt = time.time()
    b = x.shape[0]
    w = w[:, None, None, None]
    c = - torch.special.expm1(logsnr - logsnr_next)
    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (
        -logsnr).sigmoid(), (-logsnr_next).sigmoid()
    alpha, sigma, alpha_next = map(
        lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    batch = xt2batch(x=x, z=z, logsnr=logsnr.repeat(b), t=t)
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


if __name__ == '__main__':
    pass
