import numpy as np
import torch

import torch
import torch.nn.functional as F


def lat_long_to_point_xyz(lat_rad, lng_rad):
    """Convert lat/long coodtinates to a 3D XYZ point (assume unitary radio).
    
    # Parameters
        lat_rad: degree from -pi/2 to pi/2
        lng_rad: degree from -pi to pi

    # Returns
        A point xyz in the unit sphere as a list
    """
    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.sin(lat_rad)
    z = -np.cos(lat_rad) * np.sin(lng_rad)

    return [x, y, z]


def point_xyz_to_lat_long(p_xyz):
    lat = np.arctan2(p_xyz[1], np.sqrt(p_xyz[0] ** 2 + p_xyz[2] ** 2))
    lng = np.arctan2(-p_xyz[2], p_xyz[0])

    return lat, lng


def uv2cam(uv, z, intrinsics, homogeneous=False):
    """
    # Arguments
        uv: array with UV coordinates in pixel coordinates (2, H*W)
        z: Scalar value for the Z coordinate (depth)
        intrinsics: array with camera intrinsics (3, 3)
        homogeneous: whether to return in homogeneous coords

    # Returns
        An array with shape (3, H*W) or (4, H*W)
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x_lift = (uv[0] - cx) / fx * z
    y_lift = (uv[1] - cy) / fy * z
    z_lift = np.ones_like(x_lift) * z

    if homogeneous:
        return np.stack([x_lift, y_lift, z_lift, np.ones_like(z_lift)])
    else:
        return np.stack([x_lift, y_lift, z_lift])


def get_ray_direction(ray_start, intrinsics, inv_RT, uv_pix, depths=1.0):
    """
    # Arguments
       ray_start: numpy (3,)
        intrinsics: camera intrinsics (3, 3)
        inv_RT: Inverted homogeneous [R|T] matrix, camera to world projection (4, 4)
        uv_pix: UV mapping in pixels for each output ray, shape (2, h, w), where `num_rays=h*w`
        depths: scalar or None

    # Returns
        Array with ray directions with shape (num_rays, 3), where
        `num_rays = (h - y) * (w - x)`
    """
    ray_dir_cam = uv2cam(uv_pix, depths, intrinsics, True) # UV+D->XYZ homogeneous points (4, H, W)
    ray_dir_cam = ray_dir_cam.reshape(4, -1)
    ray_dir_world = (inv_RT @ ray_dir_cam)[:3] # camera to world projection (3, H*W)

    # Normalize rays
    ray_dir = ray_dir_world - ray_start[:, np.newaxis]
    ray_dir_l2 = np.linalg.norm(ray_dir, ord=2, axis=0, keepdims=True)
    ray_dir /= np.clip(ray_dir_l2, 1e-6, None)

    return ray_dir.T


def project_global_3d_to_image(pts, intrinsics, extrinsics_w2c, return_depth=False):
    """
    # Arguments
        pts: numpy (N, 3)
        intrinsics: camera intrinsics (3, 3)
        extrinsics_w2c: Homogeneous [R|T] matrix, world to camera projection (4, 4)
        return_depth: boolean, whether to return depth values or not

    # Returns
        Array with projected points into the image plane (N, 2).
        If `return_depth=True`, return (N, 3).
    """
    hom_pts = np.concatenate([pts, np.ones_like(pts[:, :1])], axis=-1)
    cam_pts = (hom_pts @ extrinsics_w2c.T)[:, :3]
    pts_uv = cam_pts[:, :2] / cam_pts[:, 2:3]
    pts_uv = (pts_uv[:, :2] @ intrinsics[:2, :2].T) + intrinsics[0:2, 2:3].T
    if return_depth:
        return np.concatenate([pts_uv, cam_pts[:, 2:3]], axis=-1)
    return pts_uv


def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2==0] = 1
        return x / l2, l2


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def compute_normal_map(ray_start, ray_dir, depths, rays_hits, size):
    """Compute normal map from depth map

    # Arguments
        ray_start: tensor (1, 3)
        ray_dir: tensor (num_rays, 3)
        depths: regressed depth values from the implicit field (num_rays,)
        rays_hits: tensor with full buffer if hitting flags, used to recover the full
            shape of the tensor from where rays where sampled, shape (N,), where `N >= num_rays`
        size: size of the image from where rays were sampled (W, H)

    # Returns
        Normals as a tensor with shape (H, W, 3) and xyz coordinates as (H, W, 3)
    """
    from .utils import masked_scatter
    W, H = size

    # this function is pytorch-only
    xyz_coords = ray(ray_start, ray_dir, depths.unsqueeze(-1)).transpose(0, 1) # world coords (3, num_rays)

    xyz_coords = masked_scatter(rays_hits, xyz_coords.T)[0] # (N, 3)
    xyz_coords_wh = xyz_coords.view(H, W, 3)

    # estimate local normal
    shift_l = xyz_coords_wh[2:,  :, :]
    shift_r = xyz_coords_wh[:-2, :, :]
    shift_u = xyz_coords_wh[:,  2:, :]
    shift_d = xyz_coords_wh[:, :-2, :]
    diff_hor = normalize(shift_r - shift_l, axis=2)[0][:, 1:-1, :]
    diff_ver = normalize(shift_u - shift_d, axis=2)[0][1:-1, :, :]
    normal = cross(diff_ver, diff_hor, axis=2)
    _normal = normal.new_zeros(*xyz_coords_wh.size())
    _normal[1:-1, 1:-1, :] = normal

    return _normal, xyz_coords_wh


def compute_normal_map2(ray_start, ray_dir, depths, size):
    """Compute normal map from depth map

    # Arguments
        ray_start: tensor (1, 3)
        ray_dir: tensor (N=H*W, 3)
        depths: regressed depth values from the implicit field (N, 1)
        size: size of the image from where rays were sampled (W, H)

    # Returns
        Normals as a tensor with shape (H, W, 3) and xyz coordinates as (H, W, 3)
    """
    W, H = size
    assert H * W == depths.size(0), (f'Error! Invalid size ({size}) for input tensor {depths.shape}')

    # this function is pytorch-only
    xyz_coords = ray(ray_start, ray_dir, depths) # world coords (N, 3)
    xyz_coords_wh = xyz_coords.view(1, H, W, 3).permute(0, 3, 1, 2) # (1, 3, H, W)

    # Pad tensor
    xyz_coords_wh_padh = F.pad(xyz_coords_wh, (0, 0, 0, 0, 1, 1), mode='reflect')
    xyz_coords_wh_padw = F.pad(xyz_coords_wh, (0, 0, 1, 1, 0, 0), mode='reflect')

    # estimate local normal
    
    diff_h = normalize(xyz_coords_wh_padh[:, 2:, :, :] - xyz_coords_wh_padh[:, :-2, :, :], axis=-1)[0]
    diff_w = normalize(xyz_coords_wh_padw[:, :, 2:, :] - xyz_coords_wh_padw[:, :, :-2, :], axis=-1)[0]

    normal = cross(diff_w, diff_h, axis=-1)
    # pool = torch.nn.AvgPool2d(2, 1)
    # normal = pool(normal.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    # normal = F.avg_pool2d(normal.permute(0, 3, 1, 2), 2, 1).permute(0, 2, 3, 1)
    normal = normalize(normal, axis=-1)[0]

    # _normal = normal.new_zeros(*xyz_coords_wh.size())
    # _normal[1:-1, 1:-1, :] = normal

    return normal, xyz_coords_wh


def cart2sph(pts_cart, captury_coord_sys=True):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cart (array_like): Cartesian :math:`x`, :math:`y` and
            :math:`z`. Of shape N-by-3 or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)


    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians.
    """
    pts_cart = np.array(pts_cart)


    # Validate inputs
    is_one_point = False
    if pts_cart.shape == (3,):
        is_one_point = True
        pts_cart = pts_cart.reshape(1, 3)
    elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Convert from Captury system
    if captury_coord_sys:
        # In captury system, x=z', y=x', z=y'
        pts_cart_aux = pts_cart.copy()
        pts_cart[:, 0] = pts_cart_aux[:, 2]
        pts_cart[:, 1] = pts_cart_aux[:, 0]
        pts_cart[:, 2] = pts_cart_aux[:, 1]


    # Compute r
    r = np.sqrt(np.sum(np.square(pts_cart), axis=1))

    # Compute latitude
    z = pts_cart[:, 2]
    lat = np.arcsin(z / r)

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    lng = np.arctan2(y, x) # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = np.stack((r, lat, lng), axis=-1)

    if is_one_point:
        pts_r_lat_lng = pts_r_lat_lng.reshape(3)

    return pts_r_lat_lng


def sph2cart(pts_sph, captury_coord_sys=True):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Convert to latitude-longitude convention, if necessary
    pts_r_lat_lng = pts_sph

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]

    if captury_coord_sys:
        x = r * np.cos(lat) * np.sin(lng) # y'
        y = r * np.sin(lat) # z'
        z = r * np.cos(lat) * np.cos(lng) # x'
    else:
        z = r * np.sin(lat)
        x = r * np.cos(lat) * np.cos(lng)
        y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def gen_light_ray_dir(envmap_h, envmap_w, envmap_radius=5000):
    # OpenEXR "latlong" format
    # lat = pi/2 (vert)
    # lng = pi (horiz)
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = 180 / (envmap_h)
    lng_step_size = 360 / (envmap_w)
    # lats = np.linspace(90 - lat_step_size / 2, -90 + lat_step_size / 2, envmap_h)
    # lngs = np.linspace(180 - lng_step_size / 2, -180 + lng_step_size / 2, envmap_w)
    lats = np.linspace(90 - lat_step_size, -90 + lat_step_size, envmap_h)
    lngs = np.linspace(180 - lng_step_size, -180 + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(np.pi * lngs / 180, np.pi * lats / 180, indexing='xy')

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs, captury_coord_sys=True).astype(np.float32) # (lat * lng, 3)
    light_dir = normalize(xyz)[0]

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat) # (lat, lng)

    # TODO: shall we consider the area of each light source or not?
    # areas_ = np.ones_like(areas)
    # areas = areas.sum() * areas_ / areas_.sum()

    return xyz.astype(np.float32), light_dir.astype(np.float32), areas.astype(np.float32)