import os

import cv2
import numpy as np

import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from opendr.geometry import VertNormals

from tex.iso import Isomapper
from tex.iso import IsoColoredRenderer

from utils import decode_str_option_filter
from utils import read_obj
from utils import read_obj_mesh
from utils import read_meshes_file
from utils import preload_crop_filedict



def generate_textures(f, vt, ft, meshes_vt, faces, cameras, img_w, img_h,
                      crop, processed_data_dir, camera_dict,
                      tex_resolution=512,
                      bgcolor=np.array([1., 0.2, 1.]),
                      only_normals=False):

    iso = Isomapper(vt, ft, faces, tex_resolution, bgcolor=bgcolor)
    iso_vis = IsoColoredRenderer(vt, ft, faces, tex_resolution)

    vn = VertNormals(f=faces, v=meshes_vt)
    normal_tex = iso_vis.render(vn / 2.0 + 0.5)

    if only_normals:
        return None, normal_tex

    best_num = 3
    tex_agg = np.zeros((tex_resolution, tex_resolution, best_num, 3))
    tex_agg[:] = np.nan
    normal_agg = np.ones((tex_resolution, tex_resolution, best_num)) * 0.2

    static_indices = np.indices((tex_resolution, tex_resolution))
    kernel = np.ones((5, 5), np.uint8)

    for cam in cameras:

        x, y, w, h, sx, sy = crop[cam]
        img_path = os.path.join(processed_data_dir, 'rgb', f'{cam:03d}', f'{f:06d}.jpg')
        mask_path = os.path.join(processed_data_dir, 'mask', f'{cam:03d}', f'{f:06d}.png')

        with Image.open(img_path) as im:
            image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            image[y : h + y, x : w + x] = np.array(im)

        with Image.open(mask_path) as im:
            mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
            cmask = cv2.erode(np.array(im), kernel)
            mask[y : h + y, x : w + x, 0] = cmask
            mask = np.tile(mask, (1, 1, 3))

        indices = np.where(np.all(mask < np.array([8, 8, 8]), axis=-1))
        mask_content = np.ones([image.shape[0], image.shape[1]]) * 255
        mask_content[indices[0], indices[1]] = 0
        mask = np.array(mask_content > 0, dtype=np.uint8)

        camera = ProjectPoints(
            t=camera_dict[cam]['camera_t'],
            rt=camera_dict[cam]['camera_rt'],
            c=camera_dict[cam]['camera_c'],
            f=camera_dict[cam]['camera_f'],
            k=camera_dict[cam]['camera_k'],
            v=meshes_vt)

        frustum = {
            'near': 100.,
            'far': 10000.,
            'width': int(camera_dict[cam]['width']),
            'height': int(camera_dict[cam]['height'])
            }

        rn_vis = ColoredRenderer(f=faces, frustum=frustum, camera=camera, num_channels=1)

        visibility = rn_vis.visibility_image.ravel()
        visible = np.nonzero(visibility != 4294967295)[0]

        proj = camera.r  # projection
        in_viewport = np.logical_and(
            np.logical_and(np.round(camera.r[:, 0]) >= 0, np.round(camera.r[:, 0]) < frustum['width']),
            np.logical_and(np.round(camera.r[:, 1]) >= 0, np.round(camera.r[:, 1]) < frustum['height']),
        )
        in_mask = np.zeros(camera.shape[0], dtype=np.bool)
        idx = np.round(proj[in_viewport][:, [1, 0]].T).astype(np.int).tolist()

        in_mask[in_viewport] = mask[idx[0], idx[1]]

        faces_in_mask = np.where(np.min(in_mask[faces], axis=1))[0]
        visible_faces = np.intersect1d(faces_in_mask, visibility[visible])

        if (visible_faces.size != 0):
            part_tex = iso.render(image / 255., camera, visible_faces)
        
            # angle under which the texels have been seen
            points = np.hstack((proj, np.ones((proj.shape[0], 1))))
            points3d = camera.unproject_points(points)
            points3d /= np.linalg.norm(points3d, axis=1).reshape(-1, 1)
            alpha = np.sum(points3d * vn.r, axis=1).reshape(-1, 1)
            alpha[alpha < 0] = 0
            iso_normals = iso_vis.render(alpha)[:, :, 0]
            iso_normals[np.all(part_tex == bgcolor, axis=2)] = 0

            # texels to consider
            part_mask = np.zeros((tex_resolution, tex_resolution))
            min_normal = np.min(normal_agg, axis=2)
            part_mask[iso_normals > min_normal] = 1.

            # update best seen texels
            where = np.argmax(np.atleast_3d(iso_normals) - normal_agg, axis=2)
            idx = np.dstack((static_indices[0], static_indices[1], where))[part_mask == 1]
            tex_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = part_tex[part_mask == 1]
            normal_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = iso_normals[part_mask == 1]

    # merge textures
    tex_median = np.nanmedian(tex_agg, axis=2)

    where = np.max(normal_agg, axis=2) > 0.2

    tex_mask = iso.iso_mask
    mask_final = np.float32(where)

    kernel_size = np.int(tex_resolution * 0.1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    inpaint_area = cv2.dilate(tex_mask, kernel) - mask_final

    tex_median = cv2.cvtColor((255 * tex_median).astype(np.uint8), cv2.COLOR_RGB2BGR)
    tex_final = cv2.inpaint(tex_median, (255 * inpaint_area).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    tex_final = cv2.cvtColor(tex_final, cv2.COLOR_BGR2RGB)

    return tex_final, normal_tex


def load_cameras(intr_path, extr_path, cameras, image_w, image_h):

    # Load camera
    camera_dict = {}
    for cam in cameras:
        intrin_camera_file = os.path.join(intr_path, f'intr_{cam:04d}.txt')
        extrin_camera_file = os.path.join(extr_path, f'extr_{cam:04d}.txt')

        intrin = np.loadtxt(intrin_camera_file)
        pose = np.loadtxt(extrin_camera_file)
        pose[:3, 3] *= 1000
        RT = np.linalg.inv(pose)
        rot = RT[:3,:3]
        trans = RT[:3, 3]
        R, J = cv2.Rodrigues(rot)
        rt_vec = R[:,0]

        camera_dict[cam] = {}
        camera_dict[cam]['camera_t'] = trans
        camera_dict[cam]['camera_rt'] = rt_vec
        camera_dict[cam]['camera_f'] = np.array([intrin[0, 0], intrin[1, 1]])
        camera_dict[cam]['camera_c'] = np.array([intrin[0, 2], intrin[1, 2]]) 
        camera_dict[cam]['camera_k'] = np.zeros(5)

        camera_dict[cam]['width'] = image_w
        camera_dict[cam]['height'] = image_h

        camera_dict[cam] = camera_dict[cam]

    return camera_dict


if __name__ == '__main__':

    # Set parameters 
    parser = argparse.ArgumentParser('Extract texture and normals from image data')
    parser.add_argument('processed_data_dir', type=str)
    parser.add_argument('smpl_tracking_dir', type=str)
    parser.add_argument('--resolution', type=int, default=512, help='Texture and normals resolution')
    parser.add_argument('--image_w', type=int, default=4112, help='Image width resolution')
    parser.add_argument('--image_h', type=int, default=3008, help='Image height resolution')
    parser.add_argument('--start_frame', type=int, default=0, help='Index of the first frame')
    parser.add_argument('--num_frames', type=int, default=999999, help='Total number of frames (auto limit to the sequence length)')
    parser.add_argument('--subsample', type=int, default=1, help='subsampling factor')
    parser.add_argument('--cameras', type=str, default="0,7,14,18,27,40", help='Cameras to be used, as a list or interval, e.g. 0..10')
    parser.add_argument('--only_normals', action="store_true", help='Compute only SMPL normals (very fast) and ignore texture')
    args = parser.parse_args()

    if args.processed_data_dir[-1] == "/":
        args.processed_data_dir = args.processed_data_dir[:-1] # remove trailing slash

    cameras = decode_str_option_filter(args.cameras)
    image_w, image_h = args.image_w, args.image_h

    smpl_path = args.smpl_tracking_dir
    assert os.path.exists(smpl_path), (f'Path with SMPL parameters not found! "{smpl_path}"')

    output_tex_dir = os.path.join(args.processed_data_dir, 'tex')
    if not os.path.exists(output_tex_dir):
        Path(output_tex_dir).mkdir(parents=True, exist_ok=True)

    output_normal_dir = os.path.join(args.processed_data_dir, 'normal')
    if not os.path.exists(output_normal_dir):
        Path(output_normal_dir).mkdir(parents=True, exist_ok=True)

    model_file = os.path.join(smpl_path, 'canonical.obj')
    vt, ft = read_obj(os.path.join(smpl_path, 'uvmapping.obj'))
    ft -= 1

    print (f'Info: Loading meshes.meshes file...')
    canonical_file = os.path.join(smpl_path, 'canonical.obj')
    _, faces = read_obj_mesh(canonical_file)
    meshes_file = os.path.join(smpl_path, 'meshes.meshes')
    meshes_vt = read_meshes_file(meshes_file)
    num_frames = len(meshes_vt)
    args.end_frame = min(num_frames, args.num_frames + args.start_frame)

    frames = list(range(args.start_frame, args.end_frame, args.subsample))
    crop_filedict = preload_crop_filedict(os.path.join(args.processed_data_dir, 'crop'), cameras, frames)

    intr_path = os.path.join(os.path.join(args.processed_data_dir), 'cameras', 'intr')
    extr_path = os.path.join(os.path.join(args.processed_data_dir), 'cameras', 'extr')
    camera_dict = load_cameras(intr_path, extr_path, cameras, image_w, image_h)

    for f in tqdm(frames):
        texture, normals = generate_textures(f, vt, ft, meshes_vt[f], faces, cameras, image_w, image_h,
                                       crop_filedict[f], args.processed_data_dir,
                                       camera_dict, tex_resolution=args.resolution,
                                       bgcolor=np.array([1., 0.2, 1.]),
                                       only_normals=args.only_normals)
    
        im_normals = Image.fromarray((255 * normals).astype(np.uint8))
        normal_filename = os.path.join(output_normal_dir, f'{f:06d}.png')
        im_normals.save(normal_filename)

        if args.only_normals is False:
            im_texture = Image.fromarray(texture)
            # texture_filename = os.path.join(output_tex_dir, f'{f:06d}.png')
            # im_texture.save(texture_filename)
            texture_filename = os.path.join(output_tex_dir, f'{f:06d}.jpg')
            im_texture.save(texture_filename, quality=97)



