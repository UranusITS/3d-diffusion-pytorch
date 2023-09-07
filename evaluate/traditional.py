import sys
sys.path.append("..")

from math import isinf, isnan
from data_process.IXIdataset import IXIDataset
from data_process.AMOSdataset import AMOSDataset
from data_process.normalization import CTNormalization
from torch.utils.data import DataLoader
from skimage.transform import resize
from tqdm import tqdm
from utils import *

batch_size = 1
input_width = 256
output_width = 256
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
    dataset = IXIDataset('/data_hdd/users/lisikuang/IXI/T1/valid/', normalization, input_width=input_width, output_width=output_width)
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
    dataset = AMOSDataset('test', normalization=normalization, input_width=input_width, output_width=output_width)
else:
    raise NotImplementedError

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16)

psnr_avg = 0
psnr_cnt = 0
ssim_avg = 0
ssim_cnt = 0

pbar = tqdm(loader)
for X, Y, T in pbar:
    X = X[0].squeeze()
    Y = Y[0].squeeze()

    pred = resize(X, (3, Y.shape[0], Y.shape[1]))[1]
    gt = normalization.rerun(Y.numpy())
    pred = normalization.rerun(pred)
    psnr = compare_psnr(gt, pred)
    ssim = compare_ssim(gt, pred, 11)
    if not isinf(psnr) and not isnan(psnr):
        psnr_avg += psnr
        psnr_cnt += 1
    if not isinf(ssim) and not isnan(ssim):
        ssim_avg += ssim
        ssim_cnt += 1
    pbar.set_postfix_str('psnr: {:.2f}  ssim: {:.2f}'.format(psnr_avg / psnr_cnt if psnr_cnt else 0.00, ssim_avg / ssim_cnt if ssim_cnt else 0.00))

psnr_avg /= psnr_cnt
ssim_avg /= ssim_cnt

print('psnr_count:', psnr_cnt)
print('psnr_avg:', psnr_avg)
print('ssim_count:', ssim_cnt)
print('ssim_avg:', ssim_avg)
