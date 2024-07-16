
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
import argparse
from PIL import Image
import imageio
from pathlib import Path
import torch

from lpips_pytorch import LPIPS
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import skimage

from .utils import decode_str_option_filter
from .utils import func_linear2srgb


def parse_input_args(input_args):
    parser = argparse.ArgumentParser(description='Evaluate Relight from EXR Files')
    parser.add_argument('-f', type=str, help='Default argument (used only for compatibility in Jupyter Lab)')

    parser.add_argument('--input', type=str, required=True, help='Input path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--mask_prefix', type=str, default='mask')
    parser.add_argument('--frames', type=str, required=True)
    parser.add_argument('--cameras', type=str, required=True)
    parser.add_argument('--rgb_file_format', type=str, default='jpg')

    parsed_args = parser.parse_args(args=input_args)

    return parsed_args

newsize = None

def pil_img_loader(fname, newsize=newsize):
    with Image.open(fname) as im:
        if newsize is not None:
            im = im.resize(newsize)
        im_arr = np.array(im)
    return (im_arr / 255.0).astype(np.float32)

def iio_loader(fname, newsize=newsize):
    im = imageio.imread(fname)
    if newsize is not None:
        im = cv2.resize(im, newsize, interpolation=cv2.INTER_AREA)
    return im.astype(np.float32)

def np_loader(fname, newsize=newsize):
    with np.load(fname) as im:
        im = im['arr_0']
    if newsize is not None:
        im = cv2.resize(im, newsize, interpolation=cv2.INTER_AREA)
    return im.astype(np.float32)

def load_images(path, frames, cameras, prefix, ext='png'):
    data = []
    if ext in ['jpg', 'png']:
        imloader = pil_img_loader
    elif ext in ['exr', 'hdr']:
        imloader = iio_loader
    elif ext in ['npz', 'np']:
        imloader = np_loader
    else:
        raise ValueError(f'Extension {ext} not supported!')

    for f in frames:
        for c in cameras:
            fname = os.path.join(path, f'{prefix}_f_{f:06d}_c_{c:03d}_b_00.' + ext)
            data.append(imloader(fname))

    return data


def optimize_linear_coef(images, mask, hdri):
    mask = mask.gt(0.5)
    ref_rgb = images[mask]
    hdr = hdri[mask]
    rgb_factor_init = np.log(np.e - 1) * np.ones((1, 3))
    rgb_factor = torch.tensor(rgb_factor_init,
                              dtype=torch.float32, device=hdri.device, requires_grad=True)
    optimizer = torch.optim.Adam([rgb_factor], lr=0.1)
    # optimizer = torch.optim.Adam([rgb_factor], lr=0.0001)
    # for iter in tqdm(range(1), desc='Optimizing linear coefficient'):
    for iter in tqdm(range(100), desc='Optimizing linear coefficient'):
        c = torch.nn.functional.softplus(rgb_factor)
        pred_rgb = func_linear2srgb(c * hdr)
        loss = torch.mean((pred_rgb - ref_rgb) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return c.detach()


def margin_and_quantize(x, x_min, x_max, quantize_block=16, offset=16):
    x = (x + offset) / quantize_block
    if offset > 0:
        x = quantize_block * np.ceil(x)
    else:
        x = quantize_block * np.floor(x)

    return max(x_min, min(x_max, x.astype('int')))

def eval_lpips_on_np(x, y, lpips_fn):
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0)
    loss = lpips_fn(x, y)

    return loss


def eval_for_camera(args, camera):
    frames = decode_str_option_filter(args.frames)
    num_optim_frames = 1

    images = load_images(args.input, frames, [camera], 'rgb', 'jpg')
    mask = load_images(args.input, frames, [camera], args.mask_prefix, 'png')
    prob_hit = load_images(args.input, frames, [camera], 'prob_hit', 'png')
    hdri = load_images(args.input, frames, [camera], 'pred_hdr', 'npz')

    rgb_factor = optimize_linear_coef(
        torch.from_numpy(np.concatenate([images[:num_optim_frames]], axis=-1)).to(device),
        torch.from_numpy(np.concatenate([mask[:num_optim_frames]], axis=-1)).to(device),
        torch.from_numpy(np.concatenate([hdri[:num_optim_frames]], axis=-1)).to(device),
    ).cpu().unsqueeze(0)

    post_fix = []
    for f in frames:
        post_fix.append(f'_f_{f:06d}_c_{camera:03}_b_00')

    agg_psnr = []
    agg_ssim = []
    agg_lpips = []

    lpips_fn = LPIPS(net_type='alex', version='0.1')

    for i in tqdm(range(len(images)), desc='Performing inference'):
        ref_im = images[i]
        pred_im = func_linear2srgb(rgb_factor * torch.from_numpy(hdri[i])).numpy()
        ref_mask = mask[i]

        # Save the final reference, prediction and mask
        fname = os.path.join(args.output, 'rgb' + post_fix[i] + '.jpg')
        Image.fromarray((255 * ref_im).astype(np.uint8)).save(fname)

        fname = os.path.join(args.output, 'pred_relight' + post_fix[i] + '.jpg')
        p = prob_hit[i][..., np.newaxis]
        masked_pred = p * pred_im + (1 - p)
        Image.fromarray((255 * masked_pred).astype(np.uint8)).save(fname)

        fname = os.path.join(args.output, 'pred_relight_black' + post_fix[i] + '.jpg')
        p = prob_hit[i][..., np.newaxis]
        masked_pred = p * pred_im
        Image.fromarray((255 * masked_pred).astype(np.uint8)).save(fname)

        fname = os.path.join(args.output, 'mask' + post_fix[i] + '.png')
        Image.fromarray((255 * ref_mask).astype(np.uint8)).save(fname)

        fname = os.path.join(args.output, 'ref_rgb' + post_fix[i] + '.jpg')
        m = ref_mask[..., np.newaxis]
        ref_rgb = m * ref_im + (1 - m)
        Image.fromarray((255 * ref_rgb).astype(np.uint8)).save(fname)

        # Crop the minimal bbox that encloses the mask
        bm = ref_mask > 0.5
        rows = np.any(bm, axis=1)
        cols = np.any(bm, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox_margin = 16
        im_h, im_w = ref_im.shape[0:2]
        rmin = margin_and_quantize(rmin, 0, im_h - 1, bbox_margin, -bbox_margin)
        rmax = margin_and_quantize(rmax, 0, im_h - 1, bbox_margin, bbox_margin)
        cmin = margin_and_quantize(cmin, 0, im_w - 1, bbox_margin, -bbox_margin)
        cmax = margin_and_quantize(cmax, 0, im_w - 1, bbox_margin, bbox_margin)
        ref_rgb = ref_rgb[rmin:rmax, cmin:cmax]
        masked_pred = masked_pred[rmin:rmax, cmin:cmax]

        # Here we compute the metrics and accumulate
        psnr = skimage.metrics.peak_signal_noise_ratio(ref_rgb, masked_pred, data_range=1)
        ssim = skimage.metrics.structural_similarity(ref_rgb, masked_pred, channel_axis=-1, data_range=1)
        lpips = eval_lpips_on_np(ref_rgb, masked_pred, lpips_fn)

        agg_psnr.append(psnr)
        agg_ssim.append(ssim)
        agg_lpips.append(lpips)

    print (f'Results camera {str(camera)}:')
    print (f'  PSNR:    {np.mean(agg_psnr):.3f}')
    print (f'  SSIM:    {np.mean(agg_ssim):.3f}')
    print (f'  LPIPS:   {np.mean(agg_lpips):.3f}')

    return np.mean(agg_psnr), np.mean(agg_ssim), np.mean(agg_lpips)


if __name__ == '__main__':
    args = parse_input_args(sys.argv[1:])
    Path(args.output).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cameras = decode_str_option_filter(args.cameras)

    agg_psnr = []
    agg_ssim = []
    agg_lpips = []

    for c in cameras:
        psnr, ssim, lpips = eval_for_camera(args, c)
        agg_psnr.append(psnr)
        agg_ssim.append(ssim)
        agg_lpips.append(lpips)

    print (f'Final result:')
    print (f'  PSNR:    {np.mean(agg_psnr):.3f}')
    print (f'  SSIM:    {np.mean(agg_ssim):.3f}')
    print (f'  LPIPS:   {np.mean(agg_lpips):.3f}')
