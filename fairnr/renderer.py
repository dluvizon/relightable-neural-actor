# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is to simulate "generator" in fairseq
"""

import os, tempfile, shutil, glob
import time
import torch
import numpy as np
import logging
import imageio

from tqdm import tqdm

from fairnr.data import trajectory, geometry, data_utils
from fairnr.data.data_utils import get_uv, parse_views
from fairnr.data.data_utils import save_visualizations
from fairnr.data.data_utils import export_visualizations
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuralRenderer(object):
    
    def __init__(self, 
                resolution="512x512", 
                frames=501, 
                speed=5,
                raymarching_steps=None,
                path_gen=None, 
                beam=10,
                at=(0,0,0),
                up=(0,1,0),
                output_dir=None,
                output_type=None,
                fps=24,
                test_camera_poses=None,
                test_camera_intrinsics=None,
                test_camera_views=None):
        # print (f'DEBUG:: NeuralRenderer.__init__ >>>>>>>>>>>>>>>>>>')
        self.frames = frames
        self.speed = speed
        self.raymarching_steps = raymarching_steps
        self.path_gen = path_gen
        
        if isinstance(resolution, str):
            self.resolution = [int(r) for r in resolution.split('x')]
        else:
            self.resolution = [resolution, resolution]

        self.beam = beam
        self.output_dir = output_dir
        self.output_type = output_type
        self.at = at
        self.up = up
        self.fps = fps

        if self.path_gen is None:
            self.path_gen = trajectory.circle()
        if self.output_type is None:
            self.output_type = ["color"]
        # print (f'DEBUG:: NeuralRenderer:: self.output_type:', self.output_type)

        if test_camera_intrinsics is not None:
            self.test_int = data_utils.load_intrinsics(test_camera_intrinsics)
        else:
            self.test_int = None

        self.test_frameids = None
        if test_camera_poses is not None:
            if os.path.isdir(test_camera_poses):
                self.test_poses = [
                    np.loadtxt(f)[None, :, :] for f in sorted(glob.glob(test_camera_poses + "/*.txt"))]
                self.test_poses = [np.linalg.inv(t) for t in self.test_poses]
                self.test_poses = np.concatenate(self.test_poses, 0)
            else:
                self.test_poses = data_utils.load_matrix(test_camera_poses)
                if self.test_poses.shape[1] == 17:
                    self.test_frameids = self.test_poses[:, -1].astype(np.int32)
                    self.test_poses = self.test_poses[:, :-1]
                self.test_poses = self.test_poses.reshape(-1, 4, 4)

            if test_camera_views is not None:
                render_views = parse_views(test_camera_views)
                self.test_poses = np.stack([self.test_poses[r] for r in render_views])

        else:
            self.test_poses = None

    def generate_rays(self, t, intrinsics, img_size, inv_RT=None, center=None):
        """Generates a set of rays ... how?? TODO
        TODO: intrinsics has batch, but inv_RT has not ?

        # Arguments
            t: a time step ? float?
            intrinsics: torch tensor with shape (batch, 4, 4)
            img_size: torch tensor [img_h, img_w, ratio_h, ratio_w] with shape (4,), where img_h,img_w is the rendered image size 
            inv_RT: torch tensor extrinsics camera->world ? camera with shape (4, 4), or None
            center: torch tensor with the center of the scene with shape (3,) ?, or None

        # Returns
            uv: torch tensor UV mapping with shape (2, img_h*img_w) and with values
                ranging from 0 to org_h/org_w, where org_h=img_h*ratio_h
            inv_RT: same as input inv_RT, or extrinsics based on center, shape (4, 4)
        """
        if inv_RT is None:
            cam_pos = torch.tensor(self.path_gen(t * self.speed / 180 * np.pi), 
                        device=intrinsics.device, dtype=intrinsics.dtype)
            if center is None:
                center = torch.tensor(list(self.at), dtype=cam_pos.dtype, device=cam_pos.device)
            self.at = center
            cam_pos = cam_pos + self.at
            cam_rot = geometry.look_at_rotation(cam_pos, at=self.at, up=self.up, inverse=True, cv=True)
            
            inv_RT = cam_pos.new_zeros(4, 4)
            inv_RT[:3, :3] = cam_rot
            inv_RT[:3, 3] = cam_pos
            inv_RT[3, 3] = 1
        # else:
            # inv_RT = torch.from_numpy(inv_RT).type_as(intrinsics)
        
        h, w, rh, rw = img_size[0], img_size[1], img_size[2], img_size[3]
        if self.test_int is not None:
            uv = torch.from_numpy(get_uv(h, w, h, w)[0]).type_as(intrinsics)
            intrinsics = self.test_int
        else:
            uv = torch.from_numpy(get_uv(h * rh, w * rw, h, w)[0]).type_as(intrinsics)
        uv = uv.reshape(2, -1)

        return uv, inv_RT

    def parse_sample(self,sample):
        if len(sample) == 1:
            return sample[0], 0, self.frames
        elif len(sample) == 2:
            return sample[0], sample[1], self.frames
        elif len(sample) == 3:
            return sample[0], sample[1], sample[2]
        else:
            raise NotImplementedError

    @torch.no_grad()    
    def generate(self, models, sample, **kwargs):
        """ Generates what?

        # Arguments
            models: list of torch models ?
            sample: dictionary, expected keys in sample
                'shape': ?
                'intrisics': torch tensor (batch, 1, 4, 4)
                'id': ?
                'joints': torch tensor (batch, 24, 3)
                'pose': torch tensor (batch, 1, 72)
                'motion': torch tensor (batch, 3, 24, 3)
                'joints_RT': torch tensor (batch, 4, 4, 24)
                'translation': torch tensor (batch, 1, 3)
                'rotation': torch tensor (batch, 3, 3)
                'tex': torch tensor texture map (batch, tex_h, tex_w, 3), tex_h=tex_w=512
                'normal': torch tensor normals (batch, 512, 512, 3) WHY ???
                'path': string
                'view': torch tensor shape (1, 1) ???
                'uv': torch tensor UV mapping shape (batch, 1, 2, img_h*img_w), values ranging from 0 to (`img_h - 1`, `img_w - 1`)
                'colors' or 'full_rgb': torch tensor image (resized) pixels with shape (batch, 1, img_h*img_w, 3)
                'alpha': torch tensor shape (batch, 1, img_h*img_w)
                'extrinsics': torch tensor shape (batch, 1, 4, 4)
                'size': torch tensor shape (batch, 1, 4), with values [img_h, img_w, ratio_h, ratio_w]
            kwargs: not used

        # Returns
        """
        # print (f'DEBUG:: NeuralRenderer.generator()')
        model = models[0]
        model.eval()
        
        # logger.info("rendering starts. {}".format(model.text))
        output_path = self.output_dir
        image_names = []
        sample, step, frames = self.parse_sample(sample)
        camera_id = os.path.splitext(os.path.basename(sample['path'][0][0]))[0]
        # print (f'DEBUG:: generate: sample.keys():', sample.keys())
        # print (f'DEBUG:: generate: self.resolution:', self.resolution)
        
        # fix the rendering size TODO: check this again
        if 'size' in sample:
            a = sample['size'][0,0,0] / self.resolution[0]
            b = sample['size'][0,0,1] / self.resolution[1]
            sample['size'][:, :, 0] /= a
            sample['size'][:, :, 1] /= b
            sample['size'][:, :, 2] *= a
            sample['size'][:, :, 3] *= b
        else:
            # print (f'DEBUG:: self.resolution[0]', self.resolution[0])
            # print (f'DEBUG:: self.resolution[1]', self.resolution[1])
            # print (f'DEBUG:: sample[intrinsics]', sample['intrinsics'])
            sample['size'] = torch.tensor([[[self.resolution[0], self.resolution[1], 1, 1]]]).type_as(sample['intrinsics'])
        
        # HACK??
        # sample['intrinsics'][:, 0,0] *= 3.2 # sample['intrinsics'][:, 0,0] * self.resolution[1] / 2 / sample['intrinsics'][:, 0,2]
        # sample['intrinsics'][:, 1,1] *= 3.2 # sample['intrinsics'][:, 1,1] * self.resolution[0] / 2 / sample['intrinsics'][:, 1,2]
        # sample['intrinsics'][:, 0,2] *= 3.2 # self.resolution[1] / 2
        # sample['intrinsics'][:, 1,2] *= 3.2 # self.resolution[0] / 2

        for shape in range(sample['shape'].size(0)):
            max_step = step + frames
            while step < max_step:
                next_step = min(step + self.beam, max_step)
                uv, inv_RT = zip(*[
                    self.generate_rays(
                        k, 
                        sample['intrinsics'][shape], 
                        sample['size'][shape, 0],
                        sample['extrinsics'][shape, 0])
                        # self.test_poses[0] if self.test_poses is not None else None, None)
                        # sample['joints'][shape].mean(0) if 'joints' in sample else None)
                    for k in range(step, next_step)
                ])
                if self.test_frameids is not None:
                    assert next_step - step == 1
                    ids = torch.tensor(self.test_frameids[step: next_step]).type_as(sample['id'])
                else:
                    ids = sample['id'][shape : shape + 1]
                if ('full_rgb' in sample) or ('colors' in sample):
                    real_images = sample['full_rgb'] if 'full_rgb' in sample else sample['colors']
                    real_images = real_images.transpose(2, 3) if real_images.size(-1) != 3 else real_images # (batch, 1, img_h*img_w, 3)
                    real_images = torch.cat([real_images[shape:shape+1] for _ in range(step, next_step)], 1) # why ??? 
                else:
                    real_images = None

                _sample = {
                    'id': ids,
                    'colors': real_images,
                    'intrinsics': sample['intrinsics'][shape : shape + 1],
                    'extrinsics': torch.stack(inv_RT, 0).unsqueeze(0),
                    'uv': torch.stack(uv, 0).unsqueeze(0),
                    'shape': sample['shape'][shape:shape+1],
                    'view': torch.arange(
                        step, next_step, 
                        device=sample['shape'].device).unsqueeze(0),
                    'size': torch.cat([sample['size'][shape:shape+1] for _ in range(step, next_step)], 1),
                    'step': step
                }
                for key in ['vertex', 'joints', 'joints_RT', 'translation', 'rotation', 'pose', 'tex']:
                    if key in sample:
                        _sample[key] = sample[key][shape:shape+1]

                
                if 'marchingcube' in self.output_type:
                    os.makedirs(os.path.join(output_path, 'mc_mesh'), exist_ok=True)
                    with open(os.path.join(output_path, 'mc_mesh', f'{step:06d}.ply'), 'wb') as fd:
                        with data_utils.GPUTimer() as timer:
                            plydata = model.export_surfaces(**_sample)
                        plydata.text = True
                        plydata.write(fd)

                else:
                    with data_utils.GPUTimer() as timer:
                        outs = model(**_sample)
                    print(f'render time {timer.sum}s')
                    
                    for k in range(step, next_step):
                        #images = model.visualize(_sample, None, 0, k-step)
                        vis_images = model.visualize(_sample, None, 0, k-step, export_raw=True)
                        image_name = "{:06d}".format(k)

                        # image_names = save_visualizations(vis_images, image_name, self.output_type, output_path, image_names)
                        nerfactor_prefix = os.path.join(output_path, camera_id, f"train_{k:03d}")
                        Path(nerfactor_prefix).mkdir(parents=True, exist_ok=True)
                        export_visualizations(vis_images, nerfactor_prefix)
                        image_names.append(nerfactor_prefix)


                        # save pose matrix
                        prefix = os.path.join(output_path, 'pose')
                        Path(prefix).mkdir(parents=True, exist_ok=True)
                        pose = self.test_poses[0] if self.test_poses is not None else inv_RT[k-step].cpu().numpy()
                        np.savetxt(os.path.join(prefix, image_name + '.txt'), pose)    

                        # prefix = os.path.join(output_path, 'intrinsics')
                        # Path(prefix).mkdir(parents=True, exist_ok=True)
                        # intrinsics = sample['intrinsics'][k-step].cpu().numpy()
                        # np.savetxt(os.path.join(prefix, image_name + '.txt'), intrinsics)    
                
                step = next_step

        step -= 1  # BUG?
        return step, image_names


    def save_images(self, output_files, steps=None, combine_output=True):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        timestamp = time.strftime('%Y-%m-%d.%H-%M-%S',time.localtime(time.time()))
        if steps is not None:
            timestamp = "step_{}.".format(steps) + timestamp
        
        if not combine_output:
            for type in self.output_type:
                images = [imageio.imread(file_path) for file_path in output_files if type in file_path] 
                # imageio.mimsave('{}/{}_{}.gif'.format(self.output_dir, type, timestamp), images, fps=self.fps)
                imageio.mimwrite('{}/{}_{}.mp4'.format(self.output_dir, type, timestamp), images, fps=self.fps, quality=8)
        else:
            images = [[imageio.imread(file_path) for file_path in output_files if type == file_path.split('/')[-2]] for type in self.output_type]
            images = [np.concatenate([images[j][i][:,:,:3] for j in range(len(images))], 1) for i in range(len(images[0]))]
            imageio.mimwrite('{}/{}_{}.mp4'.format(self.output_dir, 'full', timestamp), images, fps=self.fps, quality=8)
        
        return timestamp

    def save_videos(self, max_frame=None):
        timestamp = time.strftime('%Y-%m-%d.%H-%M-%S',time.localtime(time.time()))
        files = sorted(glob.glob(self.output_dir + '/color/*.png'))
        max_frame if max_frame is not None else len(files)
        # files = files[:max_frame]

        writer = imageio.get_writer(
            os.path.join(self.output_dir, 'full_' + timestamp + '.mp4'), fps=self.fps)
        for i in tqdm(range(len(files))):
            ims = np.concatenate([imageio.imread(files[i].replace('color', key))[:,:,:3] for key in self.output_type], 1)
            writer.append_data(ims)
        writer.close()

    def merge_videos(self, timestamps):
        logger.info("mergining mp4 files..")
        timestamp = time.strftime('%Y-%m-%d.%H-%M-%S',time.localtime(time.time()))
        writer = imageio.get_writer(
            os.path.join(self.output_dir, 'full_' + timestamp + '.mp4'), fps=self.fps)
        for timestamp in timestamps:
            tempfile = os.path.join(self.output_dir, 'full_' + timestamp + '.mp4')
            reader = imageio.get_reader(tempfile)
            for im in reader:
                writer.append_data(im)
        writer.close()
        for timestamp in timestamps:
            tempfile = os.path.join(self.output_dir, 'full_' + timestamp + '.mp4')
            os.remove(tempfile)