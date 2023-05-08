import sys
sys.path.append("..")

from math import isinf
from IXIdataset import IXIDataset
from torch.utils.data import DataLoader
from skimage.transform import resize
from tqdm import tqdm
from utils import *

batch_size = 1

dataset = IXIDataset('/data_hdd/users/lisikuang/IXI/T1/valid/', input_width=128, output_width=128)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16)
normalization = dataset.normalization

psnr_avg = 0
psnr_cnt = 0
ssim_avg = 0
ssim_cnt = 0

for X, Y, T in tqdm(loader):
    X = X[0].squeeze()
    Y = Y[0].squeeze()

    pred = resize(X, (3, Y.shape[0], Y.shape[1]))[1]
    gt = normalization.rerun(Y.numpy())
    pred = normalization.rerun(pred)
    psnr = compare_psnr(gt, pred)
    ssim = compare_ssim(gt, pred, 11)
    if not isinf(psnr):
        psnr_avg += psnr
        psnr_cnt += 1
    ssim_avg += ssim
    ssim_cnt += 1

psnr_avg /= psnr_cnt
ssim_avg /= ssim_cnt

print('psnr_count:', psnr_cnt)
print('psnr_avg:', psnr_avg)
print('ssim_count:', ssim_cnt)
print('ssim_avg:', ssim_avg)
