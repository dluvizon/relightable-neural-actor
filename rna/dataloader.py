import os
import glob

import json
import torch
import numpy as np
from collections import OrderedDict

import cv2
from tqdm import tqdm
from PIL import Image

from .io import read_matrix_from_file
from .io import read_face_uvmapping
from .io import read_obj_mesh
from .utils import decode_str_option_filter
from .utils import np_norm
from .utils import image2float
from .geometry import get_ray_direction
from .geometry import project_global_3d_to_image
from .geometry import normalize
from .geometry import cart2sph


class NeuraDataBase(torch.utils.data.Dataset):
    """Implements a base class dataset for NeuRA.

    # Arguments
        args: dictionary with data options. See `mesh_group` and `data_group`
            in `config.py` file.
        mode: string, `train` or `valid`. Depending on the mode, loads only
            the cameras and the poses that are relevant.
    """

    def __init__(self, args, mode) -> None:
        super().__init__()
        assert mode in ['train', 'valid'], (f"Unexpected mode '{mode}' ('train', 'valid')")

        self.data_root = args.data_root
        self.relight = args.relight
        self.envmap_size = args.envmap_size

        intrinsics_path = args.intrinsics
        extrinsics_path = args.extrinsics
        transform_path = args.transform
        cameras = decode_str_option_filter(getattr(args, f'{mode}_cameras'))
        frames = decode_str_option_filter(getattr(args, f'{mode}_frames'))
        self.rand_step_probs = decode_str_option_filter(args.rand_step_probs)

        self.__load_intrinsics(intrinsics_path, cameras)
        self.__load_extrinsics(extrinsics_path, cameras)
        self.__load_mesh_data(args)
        self.__load_pose_transforms(transform_path, frames)
        self.__set_rgb_filedict(cameras, frames)
        self.__set_mask_filedict(cameras, frames)
        self.__preload_crop_filedict(cameras, frames)
        self.__set_texture_filedict(frames, base_path='tex', file_ext=args.tex_file_ext)
        # self.__set_normals_path(frames, base_path='sampled_normals')
        # self.__set_visibility_path(frames, base_path='sampled_visibility')
        # self.__set_albedo_filedict(frames, base_path='neura_albedo')

        self.cameras = cameras
        self.frames = frames


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, **kargs):
        raise NotImplementedError

    def __load_intrinsics(self, files_path, cameras):
        assert os.path.isdir(files_path), (f"__load_intrinsics: Invalid '{files_path}' path")

        intrinsics = OrderedDict()
        files = glob.glob(os.path.join(files_path, '*.txt'))
        files.sort()
        intr_dict = {}
        for file in files:
            file_name_wo_ext = os.path.splitext(os.path.basename(file))[0] # expected to be '..._{cam_id:%0d}'
            cam_num = int(file_name_wo_ext.split('_')[-1])
            intr_dict[cam_num] = read_matrix_from_file(file)

        for cam in cameras:
            assert cam in intr_dict.keys(), (f'Missing intrinsics of camera {cam}!')
            intrinsics[cam] = intr_dict[cam]

        self.__sample_idx_to_camera_key = list(intrinsics.keys())
        self.intrinsics = intrinsics

    def __load_extrinsics(self, files_path, cameras):
        assert os.path.isdir(files_path), (f"__load_extrinsics: Invalid '{files_path}' path")

        extrinsics_c2w = OrderedDict() # projection camera to world
        extrinsics_w2c = OrderedDict() # projection world to camera
        files = glob.glob(os.path.join(files_path, '*.txt'))
        files.sort()
        extr_dict = {}

        for file in files:
            file_name_wo_ext = os.path.splitext(os.path.basename(file))[0] # expected to be '..._{cam_id:%0d}'
            cam_num = int(file_name_wo_ext.split('_')[-1])
            extr_dict[cam_num] = read_matrix_from_file(file)

        for cam in cameras:
            assert cam in extr_dict.keys(), (f'Missing extrinsics of camera {cam}!')
            extrinsics_c2w[cam] = extr_dict[cam]
            extrinsics_w2c[cam] = np.linalg.inv(extrinsics_c2w[cam])

        self.extrinsics_c2w = extrinsics_c2w
        self.extrinsics_w2c = extrinsics_w2c

    def __set_rgb_filedict(self, cameras, frames, base_path='rgb'):
        rgb_filedict = OrderedDict()
        for frame in frames:
            frame_str = f'{frame:06d}'
            rgb_filedict[frame] = OrderedDict()
            for cam in cameras:
                rgb_filepath = os.path.join(base_path, f'{cam:03d}', frame_str + '.jpg')
                # rgb_filepath = os.path.join(base_path, frame_str, f'image_c_{cam:03d}.png')
                rgb_filedict[frame][cam] = rgb_filepath

        self.rgb_filedict = rgb_filedict

    def __set_mask_filedict(self, cameras, frames, base_path='mask'):
        mask_filedict = OrderedDict()
        for frame in frames:
            frame_str = f'{frame:06d}'
            mask_filedict[frame] = OrderedDict()
            for cam in cameras:
                mask_filepath = os.path.join(base_path, f'{cam:03d}', frame_str + '.png')
                mask_filedict[frame][cam] = mask_filepath

        self.mask_filedict = mask_filedict

    def __preload_crop_filedict(self, cameras, frames, base_path='crop'):
        crop_filedict = OrderedDict()
        for frame in frames:
            crop_filedict[frame] = OrderedDict()
        for cam in cameras:
            crop_filepath = os.path.join(self.data_root, base_path, f'{cam:03d}.json')
            # assert os.path.isfile(crop_filepath), (f'crop file "{crop_filepath}" does not exist!')
            if os.path.isfile(crop_filepath):
                with open(crop_filepath, 'r') as fid:
                    crop_data = json.load(fid)
                for frame in crop_data.keys():
                    framei = int(frame)
                    if framei in frames:
                        x, y, w, h, sx, sy = crop_data[frame]
                        img_w = 4112
                        img_h = 3008
                        crop_filedict[framei][cam] = tuple([int(x), int(y), int(w), int(h), int(sx), int(sy), img_w, img_h])

        self.crop_filedict = crop_filedict

    def __set_texture_filedict(self, frames, base_path='tex', file_ext='png'):
        texture_filedict = OrderedDict()
        for frame in frames:
            texture_filedict[frame] = os.path.join(base_path, f'{frame:06d}.{file_ext}')

        self.texture_filedict = texture_filedict

    # def __set_normals_path(self, frames, base_path):
    #     self.normals_base_path = os.path.join(base_path)
    #     normals_filedict = OrderedDict()
    #     for frame in frames:
    #         normals_filedict[frame] = f'{frame:06d}.png'
    #     self.normals_filedict = normals_filedict

    # def __set_visibility_path(self, frames, base_path):
    #     self.visibility_base_path = os.path.join(base_path)
    #     visibility_filedict = OrderedDict()
    #     for frame in frames:
    #         visibility_filedict[frame] = f'{frame:06d}.npz'
    #     self.visibility_filedict = visibility_filedict

    # def __set_albedo_filedict(self, frames, base_path):
    #     albedo_filedict = OrderedDict()
    #     for frame in frames:
    #         albedo_filedict[frame] = os.path.join(base_path, f'{frame:06d}.png')
    #     self.albedo_filedict = albedo_filedict

    def __load_pose_transforms(self, files_path, frames):
        assert os.path.isdir(files_path), (f"__load_pose_transforms: Invalid '{files_path}' path")

        print (f'NeuraDataBase: pre-computing posed vertices...')
        pose_transforms = OrderedDict()
        files = glob.glob(os.path.join(files_path, '*.json'))
        files.sort()
        for file in files:
            file_name_wo_ext = os.path.splitext(os.path.basename(file))[0] # expected to be '{frame:%0d}'
            frame = int(file_name_wo_ext)
            if frame in frames:
                with open(file, 'r') as fip:
                    data = json.load(fip)
                    pose_transforms[frame] = {
                        'pose': np.array(data['pose'], dtype=np.float32).reshape(1, 72),
                        'glob_rotation': np.array(data['rotation'], dtype=np.float32).reshape(3, 3),
                        'glob_translation': np.array(data['translation'], dtype=np.float32).reshape(1, 3),
                        'joints_RT': np.array(data['joints_RT'], dtype=np.float32), #.reshape(-1, 4, 4),
                    }
                    pose_transforms[frame]['vertices'] = self.get_global_posed_vertices(pose_transforms[frame])

        self.__sample_idx_to_pose_key = list(pose_transforms.keys())
        self.pose_transforms = pose_transforms


    def __load_mesh_data(self, args):
        assert os.path.isfile(args.canonical_mesh), (f"__load_smpl_mesh: Invalid '{args.canonical_mesh}' path")
        assert os.path.isfile(args.transform_tpose), (f"__load_smpl_mesh: Invalid '{args.transform_tpose}' path")
        assert os.path.isfile(args.uvmapping), (f"__load_smpl_mesh: Invalid '{args.uvmapping}' path")
        assert os.path.isfile(args.skinning_weights), (f"__load_smpl_mesh: Invalid '{args.skinning_weights}' path")

        tpose_file = getattr(args, 'transform_tpose', None)
        if tpose_file is not None:
            with open(tpose_file, 'r') as fip:
                transform_tpose = json.load(fip)
            tpose_RT = np.array(transform_tpose['joints_RT'], dtype=np.float32) # (J, 4, 4)
            self.tpose_RT = tpose_RT.reshape((-1, 16)) # (J, 4, 4) -> (J, 16)
            self.tpose_RT_inv = np.linalg.inv(tpose_RT).reshape((-1, 16)) # (J, 4, 4) -> (J, 16)
        else:
            self.tpose_RT = None
            self.tpose_RT_inv = None

        self.face_uv = read_face_uvmapping(args.uvmapping) # (F, 3, 2)
        # load buffers for joint weights
        self.skinning_weights = np.loadtxt(args.skinning_weights, dtype=np.float32) # (V, J)
        self.vertices, self.faces = read_obj_mesh(args.canonical_mesh, scale=1/1000) # (V, 3), (F, 3)

        # fname_obj = './vertices_dataloader.obj'
        # with open(fname_obj, 'w') as fid:
        #     for v in self.vertices:
        #         fid.write(f'v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n')

        self.canonical_mesh_bounds = {
            'min': self.vertices.min(axis=0),
            'max': self.vertices.max(axis=0),
        }

        assert len(self.face_uv) == len(self.faces), (
            f'Invalid UV mapping {self.face_uv.shape} and canonical mesh faces {self.faces.shape}'
        )
        assert len(self.vertices) == len(self.skinning_weights), (
            f'Invalid skinning weights {self.skinning_weights.shape} and canonical mesh vertices {self.vertices.shape}'
        )
        self.vertices_homogeneous = np.concatenate(
            [self.vertices, np.ones_like(self.vertices[:, 0:1])], axis=-1) # (num_verts, 4)

        # compute the area of each triangle in the T-pose
        triangles = self.vertices[self.faces] # (num_faces, num_triangles=3, 3)
        vecs = triangles[:, 1:3] - triangles[:, 0:1]
        self.normalized_triangles_area = 0.5 * np.linalg.norm(np.cross(vecs[:, 0], vecs[:, 1]), axis=-1)
        self.normalized_triangles_area /= self.normalized_triangles_area.sum()

        # If available, load UV map mask
        uv_mask_file = getattr(args, 'uv_mask', None)
        if uv_mask_file is not None:
            with Image.open(uv_mask_file) as im:
                im = np.array(im, dtype=np.float32)
                if im.ndim == 3:
                    im = im[..., :3].max(axis=2)
                self.uv_mask = np.array(im, dtype=np.float32) / np.clip(im.max(), 1e-5, None)
        else:
            self.uv_mask = None

        # If available, load UV map mask for UV deformation
        uv_deform_mask_file = getattr(args, 'uv_deform_mask', None)
        if uv_deform_mask_file is not None:
            with Image.open(uv_deform_mask_file) as im:
                im = np.array(im, dtype=np.float32)
                if im.ndim == 3:
                    im = im[..., :3].max(axis=2)
                self.uv_deform_mask = np.array(im, dtype=np.float32) / np.clip(im.max(), 1e-5, None)
        else:
            self.uv_deform_mask = None


    def get_global_posed_vertices(self, pose_transform):
        """Given a pose transform, compute the posed vertices in global space.

        # Arguments
            pose_transform: dict with:
                'joints_RT': joints rigid transformation as numpy with shape (4, 4, J)
                'rotation': global rotation as numpy with shape (3, 3)
                'translation': global translation as numpy with shape (1, 3)

        # Return
        """
        vertices = self.vertices_homogeneous
        if self.tpose_RT_inv is not None:   # A custom T-Pose is given for this sample
            tpose_weighted_RT = (self.skinning_weights @ self.tpose_RT_inv).reshape(-1, 4, 4)   # (V, 4, 4)
            vertices = np.einsum("ncd,nd->nc", tpose_weighted_RT, vertices)

        A = pose_transform["joints_RT"].reshape((-1, 16)) # (J, 4, 4) -> (J, 16)
        weighted_RT = (self.skinning_weights @ A).reshape(-1, 4, 4) # (V, 16) -> (V, 4, 4)
        vertices = np.einsum("ncd,nd->nc", weighted_RT, vertices) # (V, 4)
        
        global_vertices = np.matmul(vertices[:, :3], pose_transform["glob_rotation"]) \
                + pose_transform["glob_translation"][0]  # (V, 3) mesh in posed global space

        return global_vertices

    def get_pose_key_from_sample_idx(self, idx):
        return self.__sample_idx_to_pose_key[idx]

    def get_camera_key_from_sample_idx(self, idx):
        return self.__sample_idx_to_camera_key[idx]

    def __load_image_frames(self, filedict, key, camera_ids=None, subsample=1, rescale=False):
        """Load all the camera views for a given frame indexed by key

        # Returns
            rgb_frames: list of RGB arrays, each array as (w, h, 3)
            crops: list of crop info, each as [x, y, w, h, step_w, step_h, img_w, img_h],
                where (x,y) is the top-left corner in the original image,
                (w, h) is the size of the cropped image (in the original image size),
                and (step_w, step_h) is the subsampling step in pixels.
        """
        files = []
        crops = []

        filedict_key = filedict[key]
        crop_filedict_key = self.crop_filedict[key]
        for cam in camera_ids:
            files.append(filedict_key[cam])
            crops.append(list(crop_filedict_key[cam]))

        frames = []
        for i, fname in enumerate(files):
            with Image.open(os.path.join(self.data_root, fname)) as im:
                im_arr = np.array(im)

                # Check if the crop is multiple of `subsample`
                h, w = im_arr.shape[0:2]
                qh = subsample * (h // subsample)
                qw = subsample * (w // subsample)
                if qh != h:
                    im_arr = im_arr[:qh]
                    crops[i][3] = subsample * (crops[i][3] // subsample)

                if qw != w:
                    im_arr = im_arr[:, :qw]
                    crops[i][2] = subsample * (crops[i][2] // subsample)

                if subsample != 1:
                    if rescale:
                        imh, imw = im_arr.shape[0:2]
                        newsize = (imw // subsample, imh // subsample)
                        im_arr = cv2.resize(im_arr, newsize, interpolation=cv2.INTER_AREA)
                    else:
                        im_arr = im_arr[::subsample, ::subsample]
                    crops[i][4] *= subsample # sub_x
                    crops[i][5] *= subsample # sub_y
                frames.append(im_arr)

        return frames, crops


    def load_rgb_frames(self, key, camera_ids=None, subsample=1, rescale=False):
        return self.__load_image_frames(self.rgb_filedict, key, camera_ids, subsample, rescale=rescale)


    def load_mask_frames(self, key, camera_ids=None, subsample=1):
        return self.__load_image_frames(self.mask_filedict, key, camera_ids, subsample)


    def load_textures(self, key):
        """Load the texture map as an RGB image
        
        # Returns
            An array with shape (tex_h, tex_w, 3)
        """
        texture_file = self.texture_filedict[key]
        with Image.open(os.path.join(self.data_root, texture_file)) as im:
            tex = np.array(im)

        return tex

    def load_normals(self, filename):
        with Image.open(os.path.join(self.data_root, filename)) as im:
            normals = np.array(im, dtype=np.float32) / 255 # from [0..255] to [0, 1], array as (H, W, 3)
            mask = (np.sum(normals, axis=-1) > 0.5).astype(np.float32)
            normals = normalize(2 * normals - 1)[0]

        return normals, mask

    def load_visibility(self, filename):
        data = np.load(os.path.join(self.data_root, filename))
        visibility = data['visibility'].astype(np.float32) / 255
        mask = (data['mask'] > 0).astype(np.float32)

        return visibility, mask

    def load_albedo(self, key):
        albedo_file = self.albedo_filedict[key]
        with Image.open(os.path.join(self.data_root, albedo_file)) as im:
            albedo = np.array(im)

        return albedo

    def compute_rays(self, crops, camera_ids=None):
        """Compute the image rays and UV maps given the cropped regions in the image.

        # Arguments
            crops: list of lists of crop info,
                each element as [x, y, w, h, step_w, step_h, img_w, img_h] in pixel values.
                Note that (w, h) are w.r.t. the original image and not to the
                output image. That means, for a (32, 32) crop with steps=2,
                (w, h) has to be provided as (64, 64). Therefore, the number
                of rays is `num_rays = (w / step_w) * (h / step_h)`

        # Returns
            A tuple with three arrays:
                rays_start: list of rays, each with size (1, 3)
                rays_dir: list of rays, each with size (num_rays, 3)
                uv_pix: list of uv mappings, each with size (2, h / step_h, w / step_w)
        """
        if camera_ids is not None:
            assert len(crops) == len(camera_ids), (
                f'Unexpected crops (len={len(crops)}) and camera_ids (len={len(camera_ids)})')
        else:
            assert len(crops) == len(self.intrinsics), (
                f'Unexpected crops (len={len(crops)}) and intrinsics (len={len(self.intrinsics)})')
            camera_ids = [int(k) for k in self.intrinsics]

        rays_start = []
        rays_dir = []
        uv_pix_list = []

        for crop, cam in zip(crops, camera_ids):
            intr = self.intrinsics[cam]
            extr = self.extrinsics_c2w[cam]

            translation = extr[:3, 3] # (3,)
            x, y, w, h, step_w, step_h, img_w, img_h = crop

            # Get the UV coords of the center of each pixel in the image
            uv_pix = 0.5 + np.flip(np.mgrid[y : y + h : step_h, x : x + w : step_w], axis=0).astype(np.float32) # (2, h, w)

            ray_dir = get_ray_direction(translation, intr, extr, uv_pix)

            rays_start.append(translation[np.newaxis])
            rays_dir.append(ray_dir)
            uv_pix_list.append(uv_pix)

        return rays_start, rays_dir, uv_pix_list


    def compute_visible_vertices(self, crops, vertices, intrinsics, extrinsics, margin_px=32):
        verts_2d = project_global_3d_to_image(vertices, intrinsics, extrinsics) # (V, 2)


        def get_verts_mask(vu, vv, u1, v1, u2, v2, margin):
            verts_u_b = (vu >= (u1 - margin)) * (vu <= (u2 + margin))
            verts_v_b = (vv >= (v1 - margin)) * (vv <= (v2 + margin))
            return verts_u_b * verts_v_b

        valid_verts = len(crops) * [None]
        for i, crop in enumerate(crops):
            verts_u, verts_v = verts_2d[:, 0], verts_2d[:, 1]
            u1, v1 = crop[0:2]
            u2 = u1 + crop[2]
            v2 = v1 + crop[3]
            margin = margin_px
            verts_mask = get_verts_mask(verts_u, verts_v, u1, v1, u2, v2, margin)
            while verts_mask.max() == False: # This is very unlikely, but could happen
                margin = 2 * margin
                verts_mask = get_verts_mask(verts_u, verts_v, u1, v1, u2, v2, margin)

            valid_verts[i] = verts_mask

        return valid_verts


    def preload_textures(self):
        textures = {}
        print (f'Preloading texture maps... ', end='')
        num_frames = len(self.pose_transforms)
        for idx in range(num_frames):
            key = self.get_pose_key_from_sample_idx(idx)
            tex = image2float(self.load_textures(key))
            if idx == 0:
                print (f'found {num_frames} texmaps with shape {tex.shape}')
            textures[key] = np.transpose(tex, (2, 0, 1))[np.newaxis]

        return textures


    def set_load_textures(self, load_textures):
        self.__load_textures_bool = load_textures


class NeuraTrainDataloader(NeuraDataBase):
    "Implements a train dataloader for NeuRA."

    def __init__(self, args) -> None:
        super().__init__(args, 'train')
        self.num_sampled_triangles = args.num_sampled_triangles
        self.num_sampled_cameras = args.num_sampled_cameras
        assert self.num_sampled_cameras <= len(self.intrinsics), (
            f'Invalid `num_sampled_cameras` {self.num_sampled_cameras} (<= {len(self.intrinsics)})')

        if args.num_iter_per_epoch > 0:
            self.num_iter_per_epoch = args.num_iter_per_epoch 
        else:
            self.num_iter_per_epoch = None
        self.crop_size = args.crop_size
        self.static_albedo = args.static_albedo
        self.use_uv_nets = args.use_uv_nets
        self.__load_textures_bool = True


    def __len__(self):
        if self.num_iter_per_epoch is not None:
            return self.num_iter_per_epoch
        return len(self.pose_transforms)

    def __getitem__(self, idx):
        if self.num_iter_per_epoch is not None:
            idx = idx % len(self.pose_transforms)
        if idx >= len(self.pose_transforms):
            raise IndexError

        data = {
            'rays_start': [],
            'rays_dir': [],
            'valid_verts': [],
            'rgb': [],
            'mask': [],
            'uv_pix': [],
            'sample': [],
        } # holds the data to be returned

        key = self.get_pose_key_from_sample_idx(idx)
        data.update(self.pose_transforms[key])

        # Sample faces from the posed mesh to define images patches for supervision
        # both with shape (num_sampled_triangles, 3)
        mesh_sampled_points, mesh_sampled_normals = \
            self.sample_points_from_mesh(data['vertices'], self.num_sampled_triangles)

        # Look for cameras facing the selected mesh triangle
        cam_t_dict = {k: rt[:3, 3] for k, rt in self.extrinsics_c2w.items()}
        cam_ids = np.array([k for k in cam_t_dict], dtype=np.int_)
        cam_t = np.array([t for t in cam_t_dict.values()], dtype=np.float32) # (num_cam, 3)
        facing_cameras = np.inner(np_norm(cam_t), np_norm(mesh_sampled_normals)) # (num_cam, num_sampled_triangles)
        facing_cameras = np.sum(facing_cameras > 0, axis=1) # (num_cam, )

        # Choose cameras that are seeing at least half the sampled points,
        # decrease the threshold if we cannot get enough cameras under this condition
        sampled_cam_thr = self.num_sampled_triangles // 2
        while True:
            selected_cams = facing_cameras >= sampled_cam_thr # (num_cam,)
            if (np.sum(selected_cams) >= self.num_sampled_cameras):
                break
            if sampled_cam_thr == 0:
                print (f'Warning! no cameras facing the selected points.')
                selected_cams = np.ones_like(selected_cams, dtype=bool)
                break
            sampled_cam_thr /= 2

        # Sample cameras
        selected_cam_ids = cam_ids[selected_cams]
        rng = np.random.default_rng()
        selected_cam_ids = rng.choice(selected_cam_ids,
            size=self.num_sampled_cameras, replace=False)
        selected_cam_ids.sort()

        # Load images from selected cameras
        rgb_frames, crops = self.load_rgb_frames(key, camera_ids=selected_cam_ids)
        mask_frames, _ = self.load_mask_frames(key, camera_ids=selected_cam_ids)

        if self.__load_textures_bool:
            data['texture'] = image2float(self.load_textures(key))

        if self.relight:
            fname = os.path.join('sampled_normals', 'dense', f'{key:06d}.png')
            data['uv_normals_dense'], data['uv_normals_dense_mask'] = self.load_normals(fname)
            fname = os.path.join('sampled_visibility', 'dense', f'{key:06d}.npz')
            data['uv_vis_dense'], data['uv_vis_dense_mask'] = self.load_visibility(fname)

            if self.use_uv_nets:
                partial_dir = ['mid', 'all'][np.random.randint(2)]
                fname = os.path.join('sampled_normals', partial_dir, f'{key:06d}.png')
                data['uv_normals_input'], data['uv_normals_input_mask'] = self.load_normals(fname)            
                fname = os.path.join('sampled_visibility', partial_dir, f'{key:06d}.npz')
                data['uv_vis_input'], data['uv_vis_input_mask'] = self.load_visibility(fname)

        # Project the sampled points into the sampled camera views
        for i, cam in enumerate(selected_cam_ids):
            proj_2d = project_global_3d_to_image(
                pts=mesh_sampled_points,
                intrinsics=self.intrinsics[cam],
                extrinsics_w2c=self.extrinsics_w2c[cam])

            rgb_patches, mask_patches, crop_patches = self.crop_image_patches(
                    proj_2d, rgb_frames[i], mask_frames[i], crops[i],
                    crop_size=self.crop_size, rand_step_probs=self.rand_step_probs)

            # Compute rays for the image frames
            rays_start, rays_dir, uv_pix = self.compute_rays(
                    crop_patches, camera_ids=len(crop_patches) * [cam])

            valid_verts = self.compute_visible_vertices(crop_patches, vertices=data['vertices'],
                                                 intrinsics=self.intrinsics[cam],
                                                 extrinsics=self.extrinsics_w2c[cam])

            data['rays_start'] += rays_start
            data['rays_dir'] += rays_dir
            data['valid_verts'] += valid_verts
            data['rgb'] += rgb_patches
            data['mask'] += mask_patches
            data['uv_pix'] += uv_pix
            data['sample'] += len(crop_patches) * [(key, cam)]

        data['rays_start'] = np.ascontiguousarray(np.stack(data['rays_start'], axis=0))
        data['rays_dir'] = np.ascontiguousarray(np.stack(data['rays_dir'], axis=0))
        data['valid_verts'] = np.ascontiguousarray(np.stack(data['valid_verts'], axis=0))
        data['rgb'] = np.ascontiguousarray(image2float(np.stack(data['rgb'], axis=0)))
        data['mask'] = np.ascontiguousarray(image2float(np.stack(data['mask'], axis=0)))
        data['uv_pix'] = np.ascontiguousarray(np.stack(data['uv_pix'], axis=0))

        return data


    def sample_points_from_mesh(self, vertices, num_samples):
        """This function performs the following operations:
            i) randomly sample `num_samples` triangles from the given mesh
            ii) compute the normal of the selected faces
            iii) use the reflection method to sample one point inside each
            selected triangle from a uniform distribution.

        # Arguments
            vertices: array with shape (V, 3)
            num_samples: integer with the number of points to be sampled

        # Returns
            Array with samples points as (num_samples, 3) and the normals of
            the selected faces.
        """
        # Samples triangles
        rng = np.random.default_rng()
        selected_cam_ids = rng.choice(len(self.faces),
            size=num_samples, replace=False, p=self.normalized_triangles_area)

        spl_triangles = vertices[self.faces[selected_cam_ids]] # (num_samples, 3, 3)
        p0 = spl_triangles[:, 0]
        p1 = spl_triangles[:, 1] - p0
        p2 = spl_triangles[:, 2] - p0

        # Compute the normals
        normals = np_norm(np.cross(p1, p2))

        # Draw points from a normal distribution
        unifd = np.random.rand(num_samples, 2)
        idx_mirror = unifd.sum(axis=1) > 1
        unifd[idx_mirror] = 1 - unifd[idx_mirror]
        p = p0 + unifd[:, 0:1] * p1 + unifd[:, 1:2] * p2
        
        return p, normals

    def _dbg_show_points_in_image(self, centers, rgb):
        """Show points projected into the image

        # Arguments
            centers: array with image points (N, 2)
            rgb: image buffer (H, W, 3)

        # Returns
            Painted image buffer.
        """
        h, w = rgb.shape[0:2]
        for p in centers:
            u = max(5, min(int(p[0]), w - 6))
            v = max(5, min(int(p[1]), h - 6))
            for i in range(v - 4, v + 5):
                for j in range(u - 4, u + 5):
                    rgb[i, j] = [0, 1, 0]

        return rgb

    def crop_image_patches(self, centers, rgb, mask, crop, crop_size,
            max_step=8, rand_step_probs=[1, 1, 1, 1, 1, 1, 1, 1]):
        """Given the image centers, crop the RGB image and mask.

        # Arguments
            centers: array with image points (N, 2)
            rgb: image buffer (h, w, 3)
            mask: mask buffer (h, w)
            crop: crop info as [x, y, w, h, step_w, step_h, img_w, img_h] relative to the original image
            crop_size: tuple with crop size as (crop_w, crop_h)

        # Returns
            rgb_patches: list of cropped images, each as (crop_size[0], crop_size[1], 3)
            mask_crops: list of cropped masks, each as (crop_size[0], crop_size[1])
            crop_info: list of cropping information, each as [x, y, w, h, step_w, step_h, img_w, img_h],
                w.r.t. to the input image.
        """
        img_h, img_w = rgb.shape[0:2]
        crop_w, crop_h = crop_size
        rgb_patches = []
        mask_patches = []
        crop_patches = []
        rng = np.random.default_rng()
        rand_step_probs = np.array(rand_step_probs, dtype=np.float32)
        rand_step_probs /= rand_step_probs.sum()
        x, y, w, h, step_x, step_y, img_w, img_h = crop # Warning: step_x and step_y should be 1 here!

        for p in centers:
            step = int(rng.choice(max_step, size=1, p=rand_step_probs) + 1) # randomly sample a step factor

            while True: # maybe we'll need to stay here and reduce the step further
                # c1 = max(0, min(int(p[0]) - step * crop_w // 2, img_w - step * crop_w))
                # r1 = max(0, min(int(p[1]) - step * crop_h // 2, img_h - step * crop_h))
                c1 = int(np.round(int(p[0]) - step * crop_w // 2))
                r1 = int(np.round(int(p[1]) - step * crop_h // 2))
                c2 = c1 + step * crop_w
                r2 = r1 + step * crop_h

                if (c2 - c1 <= w) and (r2 - r1 <= h):
                    break
                # still not fitting
                if step > 1:
                    step -= 1
                else:
                    raise ValueError("crop_image_patches: unable to fit image patch!")

            # extract patches from cropped images, here it is guaranteed that the
            # patch fits into the crop
            ir1 = r1 - y
            ir2 = r2 - y
            dr = 0
            if ir1 < 0:
                dr = -ir1
            if ir2 >= h:
                dr = h - ir2
            r1 += dr
            r2 += dr
            ir1 += dr
            ir2 += dr

            ic1 = c1 - x
            ic2 = c2 - x
            dc = 0
            if ic1 < 0:
                dc = -ic1
            if ic2 >= w:
                dc = w - ic2
            c1 += dc
            c2 += dc
            ic1 += dc
            ic2 += dc

            rgb_patches.append(rgb[ir1:ir2:step, ic1:ic2:step].copy())
            mask_patches.append(mask[ir1:ir2:step, ic1:ic2:step].copy())
            crop_patches.append([c1, r1, c2 - c1, r2 - r1, step, step, img_w, img_h])

        return rgb_patches, mask_patches, crop_patches


class NeuraValidationDataloader(NeuraDataBase):
    "Implements a validation/testing dataloader for NeuRA."

    def __init__(self, args) -> None:
        super().__init__(args, 'valid')
        self.valid_res_subsample = args.valid_res_subsample
        self.static_albedo = args.static_albedo
        self.use_uv_nets = args.use_uv_nets
        self.load_frames = args.load_frames
        self.rotating_camera = args.rotating_camera
        if self.rotating_camera:
            rotating_cam_extr = np.loadtxt('/CT/NeuralRelightableActor2/work/cameras_extr_40_round.txt')
            rotating_cam_extr = rotating_cam_extr.reshape((-1, 4, 4))
            rotating_cam_extr[:, :3, 3] /= 1000 # Convert translation from mm to meters
            self.rotating_cam_extr = rotating_cam_extr.astype(np.float32)

        if self.load_frames is False:
            assert args.image_size is not None, (
                f'If `load_frames={self.load_frames}`, image_size has to be given as (w,h)')
            self.image_size = args.image_size
        else:
            assert self.rotating_camera is False, (
                f'Invalid combination of load_frames={self.load_frames} and rotating_camera={self.rotating_camera}')


    def __len__(self):
        # In validation, each pose and each camera is a sample
        return len(self.pose_transforms) * len(self.intrinsics)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        pose_sample_idx = idx // len(self.intrinsics)
        pose_key = self.get_pose_key_from_sample_idx(pose_sample_idx)

        if self.rotating_camera:
            cam_key = self.get_camera_key_from_sample_idx(0)
        else:
            cam_sample_idx = idx - pose_sample_idx * len(self.intrinsics)
            cam_key = self.get_camera_key_from_sample_idx(cam_sample_idx)

        camera_ids = [cam_key]

        pose_transform = self.pose_transforms[pose_key]

        # Compute global posed mesh
        global_vertices = self.get_global_posed_vertices(pose_transform)

        # Load camera views, i.e., RGB images and matting masks, textures, etc.
        if self.load_frames:
            rgb_frames, crops = self.load_rgb_frames(pose_key,
                    camera_ids=camera_ids, subsample=self.valid_res_subsample,
                    rescale=True)

            mask_frames, _ = self.load_mask_frames(pose_key,
                    camera_ids=camera_ids, subsample=self.valid_res_subsample)
        else:
            rgb_frames = mask_frames = None
            w, h = self.image_size
            crops = [
                [0, 0, w, h, self.valid_res_subsample, self.valid_res_subsample, w, h]
            ]

        texture = self.load_textures(pose_key)

        if self.rotating_camera:
            assert self.load_frames is False, (f'Invalid options with rotating_camera and load_frames')

            intr = self.intrinsics[camera_ids[0]].copy()
            extr = self.rotating_cam_extr[pose_sample_idx % len(self.rotating_cam_extr)].copy()
            extr = np.linalg.inv(extr)
            # extr = self.extrinsics_c2w[camera_ids[0]].copy()

            translation = extr[:3, 3] # (3,)
            x, y, w, h, step_w, step_h, img_w, img_h = crops[0]

            # Get the UV coords of the center of each pixel in the image
            uv_pix = 0.5 + np.flip(np.mgrid[y : y + h : step_h, x : x + w : step_w], axis=0).astype(np.float32) # (2, h, w)

            ray_dir = get_ray_direction(translation, intr, extr, uv_pix)

            rays_start = [translation[np.newaxis]]
            rays_dir = [ray_dir]
            uv_pix = [uv_pix]

            c = crops[0]
            full_uv_pix = 0.5 + np.flip(np.mgrid[0 : c[7] : c[5], 0 : c[6] : c[4]], axis=0).astype(np.float32) # (2, h, w)

            intr = intr.copy()
            intr[0,0] /= 1.8
            intr[1,1] /= 1.8

            aux_ray_dir = get_ray_direction(translation, intr, extr, full_uv_pix)
            full_lat_lng = cart2sph(aux_ray_dir)[:, 1:3].astype(np.float32) # (num_pix, 2)

            # sph2cart
            # person_position = pose_transform['translation'] # (3,)
            # rad = 3
            # lat = 0
            # lng = idx * 
            # virtual_cam_pos = sph2cart()
            ## Compute the center of the person based on pose_transform['translation']
            ## Then, get the cartesian point from lat=0, lng=0, r=3, w.r.t. the person center
            ## Define a function to change lat,lng based on the frame index (pose_sample_idx)
            ## Set rays_start as the camera position
            ## Set rays_dir based on the camera intrinsics (and the camera position and person position)
            ## Set uv_pix as standard values (check the image resolution)
            ## Set full_lat_lng in the same way as rays_dir, but using focal / 1.8 (wider cam)
            

        else:
            # Compute rays for the image frames (cropped image around the person)
            rays_start, rays_dir, uv_pix = self.compute_rays(crops, camera_ids=camera_ids)

            # Compute rays for the full frame (useful for rendering new background based on env map)
            c = crops[0]
            full_uv_pix = 0.5 + np.flip(np.mgrid[0 : c[7] : c[5], 0 : c[6] : c[4]], axis=0).astype(np.float32) # (2, h, w)

            intr = self.intrinsics[camera_ids[0]].copy()
            intr[0,0] /= 1.8
            intr[1,1] /= 1.8
            extr = self.extrinsics_c2w[camera_ids[0]]
            translation = extr[:3, 3] # (3,)

            aux_ray_dir = get_ray_direction(translation, intr, extr, full_uv_pix)
            full_lat_lng = cart2sph(aux_ray_dir)[:, 1:3].astype(np.float32) # (num_pix, 2)

        data = { # remove the list indices, since we always have a single element
            'rays_start': rays_start[0],
            'rays_dir': rays_dir[0],
            'uv_pix': uv_pix[0],
            'texture': image2float(texture),
            'vertices': global_vertices,
            'glob_rotation': pose_transform['glob_rotation'],
            'glob_translation': pose_transform['glob_translation'],
            'joints_RT': pose_transform['joints_RT'],
            # 'extrinsics': self.extrinsics_c2w[cam_key],
            'sample': (pose_key, cam_key),
            'crop': crops,
            'full_lat_lng': full_lat_lng,
        }

        if rgb_frames is not None:
            data['rgb'] = image2float(rgb_frames[0])
        if mask_frames is not None:
            data['mask'] = image2float(mask_frames[0])

        if self.relight:
            fname = os.path.join('sampled_normals', 'dense', f'{pose_key:06d}.png')
            data['uv_normals_dense'], data['uv_normals_dense_mask'] = self.load_normals(fname)
            fname = os.path.join('sampled_visibility', 'dense', f'{pose_key:06d}.npz')
            data['uv_vis_dense'], data['uv_vis_dense_mask'] = self.load_visibility(fname)

            if self.use_uv_nets:
                partial_dir = 'all'
                fname = os.path.join('sampled_normals', partial_dir, f'{pose_key:06d}.png')
                data['uv_normals_input'], data['uv_normals_input_mask'] = self.load_normals(fname)            
                fname = os.path.join('sampled_visibility', partial_dir, f'{pose_key:06d}.npz')
                data['uv_vis_input'], data['uv_vis_input_mask'] = self.load_visibility(fname)

        return data


class NeuraCameraPathGenerator(torch.utils.data.Dataset):
    "Implements a generator for new camera path."

    def __init__(self, args) -> None:
        super().__init__()

        cam_id = 7
        fname = os.path.join(args.data_root, 'cameras', 'intr', f'intr_{cam_id:04d}.txt')
        self.intrinsics = read_matrix_from_file(fname)
        fname = os.path.join(args.data_root, 'cameras', 'extr', f'extr_{cam_id:04d}.txt')
        self.extrinsics = read_matrix_from_file(fname)
        self.len = 2

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {
            self.intrinsics, self.extrinsics
        }


class NeuraNormVisDataloader(torch.utils.data.Dataset):
    """Implements a dataloader for normals and visibility maps which loads
    partial maps and full maps. This dataloader is designed to train inpainting
    CNN models.

    # Arguments
        data_root: string, path to the dataset location
        frames_str: string with the frames to be used,
            for example: '0,1,2', or '0..20', or '100..200..2'.
        num_iter_per_epoch: integer, or None to use the number of frames
        load_partial: if True, loads the partial normal maps and visibility maps.
            This is useful for training NormalNet and VisibilityNet.
        load_visibility: if True, loads the visibility maps
        uv_mask_file: string, path to the UV map mask file
    """

    def __init__(self, data_root, frames_str, training,
                 num_iter_per_epoch=None,
                 load_visibility=True,
                 uv_mask_file=None) -> None:
        super().__init__()

        self.data_root = data_root
        self.frames = decode_str_option_filter(frames_str)
        self.training = training
        self.num_iter_per_epoch = num_iter_per_epoch
        self.load_visibility = load_visibility
        self.idx_to_key = list(self.frames)

        # If available, load UV map mask
        if uv_mask_file is not None:
            with Image.open(uv_mask_file) as im:
                im = np.array(im, dtype=np.float32)
                if im.ndim == 3:
                    im = im.max(axis=2)
                self.uv_mask = np.array(im, dtype=np.float32) / np.clip(im.max(), 1e-3, None)
        else:
            self.uv_mask = None

    def __len__(self):
        if self.num_iter_per_epoch is not None:
            return self.num_iter_per_epoch
        return len(self.frames)

    def __getitem__(self, idx):
        if self.num_iter_per_epoch is not None:
            idx = idx % len(self.frames)
        elif idx > len(self.frames):
            raise IndexError
        key = self.idx_to_key[idx]
        out = {}

        if self.training:
            partial_dir = ['mid', 'all'][np.random.randint(2)]
        else:
            partial_dir = 'all'

        fname = os.path.join('sampled_normals', 'dense', f'{key:06d}.png')
        out['norm_full'], out['norm_full_mask'] = self.read_normals(fname)
        fname = os.path.join('sampled_normals', partial_dir, f'{key:06d}.png')
        out['norm_partial'], out['norm_partial_mask'] = self.read_normals(fname)

        if self.load_visibility:
            fname = os.path.join('sampled_visibility', 'dense', f'{key:06d}.npz')
            out['vis_full'], out['vis_full_mask'] = self.read_visibility(fname)
            fname = os.path.join('sampled_visibility', partial_dir, f'{key:06d}.npz')
            out['vis_partial'], out['vis_partial_mask'] = self.read_visibility(fname)

        if self.uv_mask is not None:
            out['uv_mask'] = self.uv_mask > 0.5

        return out

    def read_normals(self, filename):
        with Image.open(os.path.join(self.data_root, filename)) as im:
            normals = np.array(im, dtype=np.float32) / 255 # from [0..255] to [0, 1], array as (H, W, 3)
            mask = (np.sum(normals, axis=-1) > 0.5).astype(np.float32)
            normals = normalize(2 * normals - 1)[0]

        return normals, mask

    def read_visibility(self, filename):
        data = np.load(os.path.join(self.data_root, filename))
        visibility = data['visibility'].astype(np.float32) / 255
        mask = (data['mask'] > 0).astype(np.float32)

        return visibility, mask


if __name__ == '__main__':
    from .config import ConfigContext

    mode = 'valid' # 'train'
    with ConfigContext() as args:
        print (f'Loading data from {args.data_root} in {mode} mode...')
        
        dataset = NeuraValidationDataloader(args)
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        for f, data in enumerate(tqdm(dataloader)):
            _ = data
        
        print (f'Done.')
