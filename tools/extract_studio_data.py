import os
import sys

import pathlib
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import cv2
from multiprocessing import Pool
import json
import time

import studiotools as stools
from utils import write_2d_array_to_file
from utils import extract_multiview_video_ffmpeg


def parse_input_args(input_args):
    parser = argparse.ArgumentParser(description='Extract Video Tool')

    parser.add_argument('-f', type=str, help='Default argument (used only for compatibility in Jupyter Lab)')

    parser.add_argument('--input', type=str, required=True, help='Input path')
    parser.add_argument('--output', type=str, required=True, help='Output path')

    parser.add_argument('--segmentation_folder', type=str, default='foregroundSegmentation_refined')
    parser.add_argument('--rgb_file_format', type=str, default='jpg')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=-1)
    parser.add_argument('--subsample', type=int, default=1, help="Frame subsample factor. E.g., if 10, export only one frame every 10th.")
    parser.add_argument('--start_cam', type=int, default=0, help="Set the initial camera ID (start with).")
    parser.add_argument('--num_cams', type=int, default=-1, help="Number of cameras to extract.")
    parser.add_argument('--only_calib', action='store_true')

    parsed_args = parser.parse_args(args=input_args)

    return parsed_args


def no_exception_mkdir(path):
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except:
        pass


def no_exception_remove(path):
    try:
        os.remove(path)
    except:
        print (f'Cannot remove {path}')


def margin_and_quantize(x, x_min, x_max, quantize_block=16, offset=16):
    x = (x + offset) / quantize_block
    if offset > 0:
        x = quantize_block * np.ceil(x)
    else:
        x = quantize_block * np.floor(x)

    return max(x_min, min(x_max, x.astype(np.int)))


def postprocess_one_frame(rgb_filepath, mask_filepath,
                          filter_rgb=False, filter_mask=False,
                          crop_bbox=True, bbox_margin=32):
    """This function post-process the RGB image in `rgb_filepath` using
    the mask from `mask_filepath`.

    # Arguments
        rgb_filepath: path to RGB file
        mask_filepath: path to mask file
        filter_rgb: whether to filter RGB image
        filter_mask: whether to filter mask
        crop_bbox: whether to crop both RGB and mask images based on
            content provided by the mask.

    # Return
        Nothing, but replaces the RGB and mask files, and also writes
        a new with named "[rgb_filepath].txt" containing the cropping
        information defined as: [x, y, w, h, step_w, step_h],
                where (x,y) is the top-left corner in the original image,
                (w, h) is the size of the cropped image, and
                (step_w, step_h) is the subsampling step in pixels.
    """

    with Image.open(rgb_filepath) as im:
        rgb = np.array(im, dtype=np.uint8)
        if filter_rgb:
            rgb = cv2.fastNlMeansDenoisingColored(rgb, None, 3, 3, 3, 7)
        # rgb = rgb.astype(np.float32) / 255
        im_h, im_w = rgb.shape[0:2]

    with Image.open(mask_filepath) as im:
        mask = np.array(im, dtype=np.uint8)
        if filter_mask:
            mask = cv2.bilateralFilter(mask, 7, 16, 13)
        mask[mask < 96] = 96
        mask[mask > 224] = 224
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        # mask = np.expand_dims(mask, axis=-1)
        mask = (255.9 * mask).astype(np.uint8)

    if crop_bbox:
        bm = mask > 8
        if bm.max() > 0:
            rows = np.any(bm, axis=1)
            cols = np.any(bm, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
        else:
            rmin = cmin = 0
            rmax, cmax = bm.shape[0:2]

        # add a margin to the bbox and quantize its boundary
        rmin = margin_and_quantize(rmin, 0, im_h - 1, bbox_margin, -bbox_margin)
        rmax = margin_and_quantize(rmax, 0, im_h - 1, bbox_margin, bbox_margin)
        cmin = margin_and_quantize(cmin, 0, im_w - 1, bbox_margin, -bbox_margin)
        cmax = margin_and_quantize(cmax, 0, im_w - 1, bbox_margin, bbox_margin)

        crop = [cmin, rmin, cmax - cmin, rmax - rmin, 1, 1]
        rgb = rgb[rmin : rmax, cmin : cmax]
        mask = mask[rmin : rmax, cmin : cmax]
    else:
        crop = [0, 0, rgb.shape[1], rgb.shape[0], 1, 1]

    # rgb = mask * rgb + (1 - mask)
    with Image.fromarray(rgb) as im:
        im.save(rgb_filepath, quality=98)
    with Image.fromarray(mask) as im:
        im.save(mask_filepath)
    with open(rgb_filepath + '.txt', 'w') as fid:
        fid.write(''.join([str(c) + ' ' for c in crop]) + '\n')


def __postprocess_frame(inputs):
        return postprocess_one_frame(inputs[0], inputs[1])


def postprocess_frames(rgb_path, mask_path, crop_json_path, args):
    """Warning! This function replaces the files in `rgb_path` and
    in `mask_path` by writing the postprocessed files with Pillow.
    It also removes the files that should be skipped (if `subsample!=1`).
    """
    print (79*'#' + '\n' + f'######## Checking frames from camera {cam_id}\n' + 79*'#')

    frame_list = []
    valid_frames = list(range(args.start_frame, args.num_frames, args.subsample))
    for f in tqdm(range(args.start_frame, args.start_frame + args.num_frames), desc='Postprocess frames'):
        rgb_filepath = os.path.join(rgb_path, f'{f:06d}.{args.rgb_file_format}')
        mask_filepath = os.path.join(mask_path, f'{f:06d}.png')
        if f in valid_frames:
            assert os.path.isfile(rgb_filepath), f'File {rgb_filepath} not found!'
            assert os.path.isfile(mask_filepath), f'File {mask_filepath} not found!'
            frame_list.append((rgb_filepath, mask_filepath))

        else:
            # remove rgb and mask
            no_exception_remove(rgb_filepath)
            no_exception_remove(mask_filepath)

    print (79*'#' + '\n' + f'######## Filtering frames from camera {cam_id}\n' + 79*'#')
    print (f'frame_list: ', len(frame_list), frame_list[0])
    with Pool(4) as p:
        r = list(tqdm(p.imap(__postprocess_frame, frame_list), total=len(frame_list)))

    # Aggregates all the image and crop information into a single json file
    cam = None
    crop_data = {}
    for rgb_path, _ in frame_list:
        txt_path = rgb_path + '.txt'
        with open(txt_path, 'r') as fid:
            line = fid.readline()
            crop = [v for v in line.strip().split()]
        aux = os.path.splitext(rgb_path)[0].split('/')
        _cam = int(aux[-2])
        frame = int(aux[-1])
        if cam is None:
            cam = _cam
        assert cam == _cam, (
            f'postprocess_frames: detected different cameras ({cam} != {_cam})'
        )
        crop_data[frame] = crop

        # now remove the file!
        no_exception_remove(txt_path)

    with open(crop_json_path, 'w') as fid:
        fid.write(json.dumps(crop_data))


if __name__ == '__main__':
    input_args = sys.argv[1:] if len(sys.argv) > 1 else None
    args = parse_input_args(input_args)

    # Read camera calib and settings files
    print (f"Info: reading config files...")
    settings = stools.read_settings_txt(os.path.join(args.input, 'settings.txt'))

    camera_calib_file = os.path.join(args.input, 'cameras.calib')
    if not os.path.isfile(camera_calib_file):
        camera_calib_file = os.path.join(args.input, 'camera.calib')
    assert os.path.isfile(camera_calib_file), (f'Error! File "{camera_calib_file}" not found!')
    cameras_txt = stools.read_camera_calib(camera_calib_file)

    start_frame = settings['START_FRAMES']
    end_frame = settings['END_FRAMES']
    K, RT = stools.compute_camera_K_and_RT(cameras_txt, settings)

    if (args.num_frames == -1) or (args.num_frames > end_frame):
        args.num_frames = end_frame - args.start_frame
    if (args.num_cams == -1) or (args.num_cams + args.start_cam > len(cameras_txt)):
        args.num_cams = len(cameras_txt) - args.start_cam

    # Create the output folders to hold the extracted multiview frames
    print (f"Info: creating folders for extracted multiview data...")
    rgb_path = os.path.join(args.output, 'rgb')
    mask_path = os.path.join(args.output, 'mask')
    crop_path = os.path.join(args.output, 'crop')
    intrinsic_path = os.path.join(args.output, 'cameras', 'intr')
    extrinsic_path = os.path.join(args.output, 'cameras', 'extr')

    no_exception_mkdir(intrinsic_path)
    no_exception_mkdir(extrinsic_path)
    if args.only_calib == False:
        no_exception_mkdir(crop_path)

    # For each camera, runs ffmpeg
    for cam_id in range(args.start_cam, args.num_cams + args.start_cam):
        print (79*'#' + '\n' + f'######## Extracting videos from camera {cam_id}\n' + 79*'#')

        # Write intrinsics and extrinsics (camera pose) to txt files
        write_2d_array_to_file(os.path.join(intrinsic_path, f'intr_{cam_id:04d}.txt'), K[cam_id])
        write_2d_array_to_file(os.path.join(extrinsic_path, f'extr_{cam_id:04d}.txt'), RT[cam_id])
        if args.only_calib:
            continue

        # Handle two options of stream{cam}.mp4 format
        input_rgb_mp4 = os.path.join(args.input, f'stream{cam_id:03d}.mp4')
        if not os.path.isfile(input_rgb_mp4):
            input_rgb_mp4 = os.path.join(args.input, f'stream{cam_id:02d}.mp4')
        if not os.path.isfile(input_rgb_mp4):
            input_rgb_mp4 = os.path.join(args.input, f'stream{cam_id:03d}.avi')
        if not os.path.isfile(input_rgb_mp4):
            input_rgb_mp4 = os.path.join(args.input, f'stream{cam_id:02d}.avi')
        assert os.path.isfile(input_rgb_mp4), (f'Error! File "{input_rgb_mp4}" not found!')

        output_rgb_path = os.path.join(rgb_path, f'{cam_id:03d}')
        while os.path.isdir(output_rgb_path) is False:
            print(f'mkdir path "{output_rgb_path}"')
            no_exception_mkdir(output_rgb_path)
            time.sleep(5) # NFS can take a time to propagate it...

        output_rgb_jpg = os.path.join(output_rgb_path, f'%06d.{args.rgb_file_format}')
        extract_multiview_video_ffmpeg(input_rgb_mp4, output_rgb_jpg, args.start_frame, args.num_frames, qscale=2)

        # Handle two options of stream{cam}.mp4 format
        input_mask_mp4 = os.path.join(args.input, args.segmentation_folder, f'stream{cam_id:03d}.mp4')
        if not os.path.isfile(input_mask_mp4):
            input_mask_mp4 = os.path.join(args.input, args.segmentation_folder, f'stream{cam_id:02d}.mp4')
        if not os.path.isfile(input_mask_mp4):
            input_mask_mp4 = os.path.join(args.input, args.segmentation_folder, f'stream{cam_id:03d}.avi')
        if not os.path.isfile(input_mask_mp4):
            input_mask_mp4 = os.path.join(args.input, args.segmentation_folder, f'stream{cam_id:02d}.avi')

        if os.path.isfile(input_mask_mp4):
            output_mask_path =os.path.join(mask_path, f'{cam_id:03d}')
            while os.path.isdir(output_mask_path) is False:
                sys.stderr.write(f'mkdir path "{output_mask_path}"')
                no_exception_mkdir(output_mask_path)
                time.sleep(5)

            output_mask_jpg = os.path.join(output_mask_path, f'%06d.png')
            extract_multiview_video_ffmpeg(input_mask_mp4, output_mask_jpg, args.start_frame, args.num_frames, monochrome=True, qscale=2)

            crop_json_path = os.path.join(crop_path, f'{cam_id:03d}.json')
            postprocess_frames(output_rgb_path, output_mask_path, crop_json_path, args)

        else:
            print (79*'#' + '\n' + f'######## Warning! No segmentation found!\n' + 79*'#')
