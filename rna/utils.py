import sys

from threading import Thread
from threading import Event
from threading import Lock
import numpy as np
import torch
import cv2

try:
    import nvidia_smi
except ImportError:
    sys.stderr.write(f"Cannot `import nvidia_smi`, disabling Tensorboard_NVidiaSMI\n")
    nvidia_smi = None

import matplotlib.pyplot as plt

from .geometry import lat_long_to_point_xyz
from .geometry import normalize as normalize_fn


def image2float(x):
    """Converts the input image from [0..255] to [0..1].
    """
    return x.astype(np.float32) / 255

def float2image(x, clip=False):
    """Converts the input array from [0..1] to [0..255].
    """
    if clip:
        x = np.clip(x, 0, 1)

    return (x * 255).astype(np.uint8)


def process_multiexposure_pictures(filepath_list, crop, new_size=None, align_images=True):
    image_jpg_list = []
    for i in range(len(filepath_list)):
        im = cv2.imread(filepath_list[i])
        im = im[crop[1]:crop[3], crop[0]:crop[2]]
        print (f"Image {filepath_list[i]}, shape:", im.shape, "max val:", im.max())
        image_jpg_list.append(im)

    if align_images:
        ## Align images
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(image_jpg_list, image_jpg_list)

    ## Resize aligned images
    image_list = []
    for im in image_jpg_list:
        h, w = im.shape[0:2]
        im = im[2:h-2, 2:w-2]
        if new_size is not None:
            im = cv2.resize(im, new_size, interpolation=cv2.INTER_AREA)
        image_list.append(im)

    return image_list


def combine_images_to_hdri(image_list, exposure_times, crf_debevec=None, debevec_samples=70, debevec_lambda=10):
    merge_debevec = cv2.createMergeDebevec()
    cal_debevec = cv2.createCalibrateDebevec(samples=debevec_samples, lambda_=debevec_lambda)
    if crf_debevec is None:
        crf_debevec = cal_debevec.process(image_list, times=exposure_times)
    hdr_debevec = merge_debevec.process(image_list, times=exposure_times.copy(), response=crf_debevec.copy())
    
    # Plot the camera response function
    # plt.rcParams["figure.figsize"] = (8, 4)
    # plt.plot(crf_debevec[:, 0, 0], color='b')
    # plt.plot(crf_debevec[:, 0, 1], color='g')
    # plt.plot(crf_debevec[:, 0, 2], color='r')
    # plt.xlabel("Measured Intensity (uint8)")
    # plt.ylabel("Calibrated Intensity (float)")
    # plt.grid()

    return hdr_debevec, crf_debevec


def compute_envmap_from_sphere_probe(hdr_image, num_lat, num_lng, glob_lat=0, glob_lng=0, glob_roll=0):
    lat_deg = np.linspace(90 * (1 - 1 / num_lat), 90 * (1 / num_lat - 1), num=num_lat)
    lng_deg = np.linspace(180 * (1 - 1 / num_lng), 180 * (1 / num_lng - 1), num=num_lng)
    
    lat_rad, lng_rad = np.meshgrid(np.pi * lat_deg / 180, np.pi * lng_deg / 180, indexing='ij')
    lat_rad += np.pi * glob_lat / 180
    lng_rad += np.pi * glob_lng / 180
    x, y, z = lat_long_to_point_xyz(lat_rad, lng_rad)
    
    glob_roll = np.pi * glob_roll / 180
    z = z * np.cos(glob_roll) - y * np.sin(glob_roll)
    y = z * np.sin(glob_roll) + y * np.cos(glob_roll)

    mag = np.sqrt((x + 1)**2 + y**2 + z**2)
    u = z / np.clip(mag, 1e-3, None) # [-1, 1]
    v = y / np.clip(mag, 1e-3, None) # [-1, 1]
    mask = mag > 1e-3
    # mask *= np.abs(x) < 0.8

    W, H = hdr_image.shape[0:2]
    u_px = (W * (1 - u) / 2).astype(np.float32)
    v_px = (H * (1 - v) / 2).astype(np.float32)
    envmap = cv2.remap(hdr_image, u_px, v_px, interpolation=cv2.INTER_NEAREST)
    
    return envmap, mask[..., np.newaxis]


def plot_crf(crf_debevec):
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.plot(crf_debevec[:, 0, 0], color='b')
    plt.plot(crf_debevec[:, 0, 1], color='g')
    plt.plot(crf_debevec[:, 0, 2], color='r')
    plt.xlabel("Measured Intensity (uint8)")
    plt.ylabel("Calibrated Intensity (float)")
    plt.grid()


def decode_str_option_filter(filter):
    f = filter.split('..')
    if len(f) == 1: # format [x,y,z]
        return [int(v) for v in f[0].split(',')]
    elif len(f) == 2: # format [x..y]
        return list(range(int(f[0]), int(f[1])))
    elif len(f) == 3: # format [x..y..z]
        return list(range(int(f[0]), int(f[1]), int(f[2])))
    else:
        raise ValueError


def decode_str_option_filter_start_end(filter):
    f = filter.split('..')
    if len(f) == 1: # format [x,y]
        s = f[0].split(',')
        return int(s[0]), int(s[1])
    elif len(f) == 2: # format [x..y]
        return int(f[0]), int(f[1])
    else:
        raise ValueError


def np_norm(vec, eps=1e-6):
    return vec / np.clip(np.linalg.norm(vec, axis=-1, keepdims=True), eps, None)


def masked_scatter(mask, src):
    """Copies elements from src to a new tensor where mask is True.

    # Arguments
        mask: tensor with shape (B, K)
        src: tensor with shape (N,) or (N, channels)

    # Returns
        A new tensor with shape (B, K), or (B, K, channels).
    """
    B, K = mask.size()
    if src.dim() == 1:
        return src.new_zeros(B, K).masked_scatter(mask, src)
    return src.new_zeros(B, K, src.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, src.size(-1)), src)



def masked_scatter_value(mask, src, value):
    """Copies elements from src to a new tensor where mask is True.

    # Arguments
        mask: tensor with shape (B, K)
        src: tensor with shape (N,) or (N, channels)

    # Returns
        A new tensor with shape (B, K), or (B, K, channels).
    """
    B, K = mask.size()
    if src.dim() == 1:
        x = value * src.new_ones(B, K)
        return x.masked_scatter(mask, src)
    x = value * src.new_ones(B, K, src.size(-1))
    return x.masked_scatter(mask.unsqueeze(-1).expand(B, K, src.size(-1)), src)


def uv_grid_to_uv_idx(grids, w, h):
    """ From UV grids as input, return the index to the UV map (nearest).
    # Arguments
        grids: tensor with shape (N, 2)
        w, h: UV image size

    # Return
        Tensor with shape (N,) with the UV index, as if UV map is a (h, w) image.
    """
    u = torch.round((w / 2) * (grids[:, 0] + 1))
    v = torch.round((h / 2) * (grids[:, 1] + 1))
    u = torch.clamp(u, 0, w - 1).long()
    v = torch.clamp(v, 0, h - 1).long()
    return u + w * v # (N,)


def infill_values(x, mask, filter_size, metric='median', valid_mask=None):
    """For all the values marked as zero in the mask, fill-in based on the
    given metric on the neighbor pixels, considering the given filter size.
    If all neighbors of i are masked out, do not update x and mask i.
    Optionally takes into account only valid pixels, given by `valid_mask`.

    # Arguments
        x: tensor with shape [H, W, ...]
        mask: tensor with shape [H, W]
        filter_size: integer size of a square filter
        metric: string

    # Returns
        A new x and mask tensors with updated values.
    """
    assert x.shape[0:2] == mask.shape, (
        f'Error: invalid x/mask shapes {x.shape}/{mask.shape}')
    assert filter_size > 1, f'Error: invalid filter size {filter_size}, must be > 1'
    valid_metrics = ['median', 'mean', 'max', 'min']
    assert metric in valid_metrics, (
        f'Error: invalid metric {metric}. Valid metrics are: ' + str(valid_metrics))

    fm = getattr(np, metric)
    nx = x.copy()
    nmask = mask.copy()
    if valid_mask is None:
        valid_mask = np.ones_like(nmask)

    num_rows, num_cols = nx.shape[0:2]
    k = filter_size // 2
    for row in range(num_rows):
        for col in range(num_cols):
            if mask[row, col] or (valid_mask[row, col] == 0):
                continue

            min_r = max(0, row - k)
            max_r = min(num_rows, row + k + 1)
            min_c = max(0, col - k)
            max_c = min(num_cols, col + k + 1)
            v = nx[min_r : max_r, min_c : max_c]
            m = mask[min_r : max_r, min_c : max_c]
            midx = m > 0
            if midx.any():
                vlist = v[midx, ...]
                nx[row, col] = fm(vlist, axis=0)
                nmask[row, col] = 1

    return nx, nmask


def func_linear2srgb(tensor):
    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor = torch.clip(tensor, 0., 1.)

    tensor_linear = tensor * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (torch.pow(tensor + 1e-7, 1 / srgb_exponent)) - (srgb_exponential_coeff - 1)

    is_linear = tensor <= srgb_linear_thres
    tensor_srgb = torch.where(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb


def func_inverse_sigmoid_np(x, eps=1e-6):
    return -np.log(1.0 / np.clip(x, eps, 1 - eps) - 1)


def compute_barycentric_coordinates(pts, triangles, min_triangle_area=1e-6):
    """This function projects each point to a single triangle and computes the
    barycentric coordinate.

    # Arguments
        pts: tensor with shape (N, 3)
        triangles: tensor with shape (N, 3, 3)

    # Returns
        The barycentric coordinates as w0, w1, w2, all with shape (N,), and
        the sign indicating inside (positive) or outside the triangle face as (N,)
    """
    # Compute the triangle normals
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]

    # Calculate the vectors from the first vertex of the triangle to the point
    to_point = pts - triangles[:, 0] # (N, 3)

    dot00 = torch.einsum("nd,nd->n", edge1, edge1)
    dot01 = torch.einsum("nd,nd->n", edge1, edge2)
    dot02 = torch.einsum("nd,nd->n", edge1, to_point)
    dot11 = torch.einsum("nd,nd->n", edge2, edge2)
    dot12 = torch.einsum("nd,nd->n", edge2, to_point)

    # Calculate the denominator of the barycentric coordinate formulas.
    denominator = dot00 * dot11 - dot01 * dot01

    # Calculate the barycentric coordinates (u, v).
    # u = torch.clamp((dot11 * dot02 - dot01 * dot12) / torch.clamp(denominator, 1e-8), 0, 1)
    # v = torch.clamp((dot00 * dot12 - dot01 * dot02) / torch.clamp(denominator, 1e-8), torch.zeros_like(u), 1 - u)
    u = (dot11 * dot02 - dot01 * dot12) / torch.clamp(denominator, 1e-8)
    v = (dot00 * dot12 - dot01 * dot02) / torch.clamp(denominator, 1e-8)
    w = 1.0 - u - v

    normals = torch.cross(edge1, edge2, dim=-1) # (N, 3)
    triangle_2area = normals.norm(dim=1)
    insideout = (to_point * normals).sum(-1).sign() #  [-1, 1] (N,)
    insideout = insideout * triangle_2area.ge(2 * min_triangle_area).float()

    return u, v, w, insideout



def post_process_uv_geometry(uv_normals, uv_vis, uv_acc,
                             uv_mask, tex_size,
                             uv_tex=None, uv_tex_cnt=None,
                             infill=False, infill_ksize=5, median_blur=False):
    """
    # Arguments
        uv_normals: tensor with shape (tex_h * tex_w, 3)
        uv_tex: tensor with shape (tex_h * tex_w, 3)
        uv_tex_cnt: tensor with shape (tex_h * tex_w)
        uv_vis: tensor with shape (tex_h * tex_w // 4, num_env_pix)
        uv_acc: tensor with shape (tex_h * tex_w // 4)
        uv_mask: tensor with shape (tex_h * tex_w)
        tex_size: tuple with (tex_w, tex_h)
        infill_ksize: integer

    # Returns
        This function returns the post-processed data as numpy arrays:
            uv_normals: shape (tex_h, tex_w, 3)
            uv_normals_mask: shape (tex_h, tex_w)
            uv_vis: shape (tex_h, tex_w, num_env_pix)
            uv_vis_mask: shape (tex_h, tex_w)
    """
    tex_w, tex_h = tex_size
    def _get_numpy_data(x, w, h, ch=None):
        if isinstance(x, torch.Tensor):
            if ch is not None:
                return x.view(h, w, ch).detach().cpu().numpy()
            else:
                return x.view(h, w).detach().cpu().numpy()
        else:
            if ch is not None:
                return np.reshape(x, (h, w, ch))
            else:
                return np.reshape(x, (h, w))

    def _process_data(data, acc, valid_mask, normalize=False):
        data = data.copy()
        acc = acc.copy()
        if infill:
            while (acc[valid_mask > 0].min() < 1):
                data, acc = infill_values(data, acc, filter_size=infill_ksize, valid_mask=valid_mask)

        if median_blur:
            data, _ = infill_values(data, valid_mask, filter_size=3)
            for ch in range(data.shape[-1]):
                data[..., ch] = cv2.medianBlur(data[..., ch], 3)
            # TODO: apply median blur when loading the data (or just let it to the cnn)

        mask = (valid_mask * acc > 0.5).astype(np.float32)
        if normalize:
            data = normalize_fn(data, axis=-1)[0]
        data = data * mask[..., np.newaxis]

        return data, mask

    uv_normals = _get_numpy_data(uv_normals, tex_w, tex_h, 3)
    # uv_tex = _get_numpy_data(uv_tex, tex_w, tex_h, 3)
    # uv_tex_cnt = _get_numpy_data(uv_tex_cnt, tex_w, tex_h)
    uv_vis = _get_numpy_data(uv_vis, tex_w // 2, tex_h // 2, -1)
    uv_acc = _get_numpy_data(uv_acc, tex_w // 2, tex_h // 2)
    uv_valid_mask = _get_numpy_data(uv_mask, tex_w, tex_h)

    uv_acc_n = np.linalg.norm(uv_normals, axis=-1) > 0.5
    uv_valid_mask_n = cv2.resize(uv_valid_mask, (tex_w, tex_h)) > 0.5
    uv_normals, uv_normals_mask = _process_data(uv_normals, uv_acc_n, uv_valid_mask_n, normalize=True)

    # uv_tex /= np.clip(uv_tex_cnt[..., np.newaxis], 1e-3, None)
    # uv_tex, uv_tex_cnt = _process_data(uv_tex, uv_tex_cnt, uv_valid_mask_n, normalize=False)

    uv_valid_mask_v = cv2.resize(uv_valid_mask, (tex_w // 2, tex_h // 2)) > 0.5
    uv_vis /= np.clip(uv_acc[..., np.newaxis], 1e-3, None)
    uv_lights = np.mean(uv_vis, axis=2)
    # mask out pixels that are too bright or too dark
    uv_acc = uv_acc * (uv_lights > 0.2) * (uv_lights < 0.5)
    uv_vis, uv_acc = _process_data(uv_vis, uv_acc, uv_valid_mask_v)

    uv_tex = None
    uv_tex_cnt = None

    return uv_normals, uv_normals_mask, uv_tex, uv_tex_cnt, uv_vis, uv_acc


class Tensorboard_NVidiaSMI(Thread):
    def __init__(self, watch_time=10):
        Thread.__init__(self)
        self.stop = Event()
        self.watch_time = watch_time
        if nvidia_smi is not None:
            nvidia_smi.nvmlInit()
            self.device_count = nvidia_smi.nvmlDeviceGetCount()
        else:
            self.device_count = 0
        self.memory_io = self.device_count * [0]
        self.util_gpu = self.device_count * [0]
        self.used_mem = self.device_count * [0]
        self.lock = Lock()

    def run(self):
        while not self.stop.wait(self.watch_time):
            with self.lock:
                for i in range(self.device_count):
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                    util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    self.memory_io[i] = util.memory
                    self.util_gpu[i] = util.gpu
                    self.used_mem[i] = mem_info.used

    def get_info(self):
        with self.lock:
            util_gpu = self.util_gpu.copy()
            used_mem = self.used_mem.copy()
            memory_io = self.memory_io.copy()
        return util_gpu, used_mem, memory_io


class UVTextures(object):
    def __init__(self, frames, size) -> None:
        self.w, self.h = size
        self.tex_buffers_rgb = {}
        self.tex_buffers_mask = {}
        for f in frames:
            self.tex_buffers_rgb[f] = 0.5 * torch.ones((self.h * self.w, 3))
            self.tex_buffers_mask[f] = torch.zeros((self.h * self.w), dtype=torch.bool)

    def update_uv_textures(self, prob_hit, grids, rgb, mask, frame):
        """
        # Arguments
            prob_hit: Tensor with shape (B, H, W)
            grids: Tensor with shape (B, H, W, 2)
            rgb: Tensor with shape (B, H, W, 3)
            mask: Tensor with shape (B, H, W)
            frame: integer
        """
        prob_hit = prob_hit.view(-1,)
        mask = mask.view(-1,)
        sel = prob_hit.gt(0.9) * mask.gt(0.9) # (N,)
        sel_grids = grids.view(-1, 2)[sel]
        sel_rgb = rgb.view(-1, 3)[sel].detach().cpu()
        idx = uv_grid_to_uv_idx(sel_grids, self.w, self.h).detach().cpu()

        tex_cache = self.tex_buffers_rgb[frame][idx]
        self.tex_buffers_rgb[frame][idx] = (sel_rgb + tex_cache) / 2
        self.tex_buffers_mask[frame][idx] = True


    def postprocess_uv_textures(self):
        for f in self.tex_buffers_rgb.keys():
            rgb = self.tex_buffers_rgb[f].view(self.h, self.w, 3).numpy().copy()
            mask = self.tex_buffers_mask[f].view(self.h, self.w).numpy().copy()
            iter = 0
            ksize = 3
            while (mask.min() < 1):
                iter += 1
                if iter > 2:
                    ksize += 2
                rgb, mask = infill_values(rgb, mask, filter_size=ksize)
            # rgb = cv2.medianBlur(rgb, 3)
            self.tex_buffers_rgb[f] = self.tex_buffers_rgb[f].new(rgb.reshape(-1, 3))


    def save_uv_textures(self, path):
        from .io import save_image
        for f in self.tex_buffers_rgb.keys():
            tex_rgb = self.tex_buffers_rgb[f].view(self.h, self.w, 3)
            tex_rgb = torch.flip(tex_rgb, dims=[0])
            save_image(os.path.join(path, f'frame_{f:06d}.png'), tex_rgb, normalize=False)


    def get_texture(self, frame, flip=True):
        tex = self.tex_buffers_rgb[frame].view(self.h, self.w, 3)
        if flip:
            return torch.flip(tex, dims=[0])
        return tex