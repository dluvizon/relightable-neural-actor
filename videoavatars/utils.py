import os
import numpy as np
import cv2
import json
from collections import OrderedDict


def read_obj_mesh(filename, scale=1.0):
    """Read an obj file and return the vertices and faces from it.
    For additional details about obj file format, please refer to
    http://paulbourke.net/dataformats/obj/
    """
    v = []
    f = []
    for content in open(filename):
        line = content.strip().split(' ')
        if line[0] == 'v': # expected 'v x y z'
            v.append([float(a) for a in line[1:]])
        if line[0] == 'f': # expected 'f v v v'
            f.append([int(a) for a in line[1:] if a])

    return scale * np.array(v, dtype=np.float32), np.array(f, dtype=int) - 1


def read_meshes_file(filename, scale=1.0):
    meshes_vt = []
    for i, content in enumerate(open(filename)):
        if i > 0: # skip header
            line = content.strip().split(' ')
            vt = np.array(line, dtype=np.float32).reshape(-1, 3)
            meshes_vt.append(vt)

    return scale * np.stack(meshes_vt, axis=0)


def read_obj(filename):
    vt, ft = [], []
    for content in open(filename):
        contents = content.strip().split(' ')
        if contents[0] == 'vt':
            vt.append([float(a) for a in contents[1:]])
        if contents[0] == 'f':
            ft.append([int(a.split('/')[1]) for a in contents[1:] if a])
    return np.array(vt, dtype='float64'), np.array(ft, dtype='int32')


def read_obj_track(filename):
    v, f = [], []
    for content in open(filename):
        contents = content.strip().split(' ')
        if contents[0] == 'v':
            v.append([float(a)*1000 for a in contents[1:]])
        if contents[0] == 'f':
            f.append([int(a) for a in contents[1:]])
    return np.array(v, dtype='float32'), np.array(f, dtype='int32')


def read_off(filename):
    with open(filename) as off:
        off.readline()
        n_v, n_f, _ = [int(a) for a in off.readline().strip().split()]
        v, f = [], []

        for i in range(n_v):
            v += [[float(a) for a in off.readline().strip().split()]]

        for i in range(n_f):
            f += [[int(a) for a in off.readline().strip().split()[1:]]]

    return np.array(v, dtype='float32'), np.array(f, dtype='int32')


def read_mask(filename):
    return np.array(cv2.imread(filename, 0) > 0, dtype=np.uint8)


def read_real(filename):
    return cv2.imread(filename, 1)


def preload_crop_filedict(path, cameras, frames):
    crop_filedict = OrderedDict()
    for frame in frames:
        crop_filedict[frame] = OrderedDict()
    for cam in cameras:
        crop_filepath = os.path.join(path, f'{cam:03d}.json')
        assert os.path.isfile(crop_filepath), (f'crop file "{crop_filepath}" does not exist!')
        with open(crop_filepath, 'r') as fid:
            crop_data = json.load(fid)
        for frame in crop_data.keys():
            framei = int(frame)
            if framei in frames:
                x, y, w, h, sx, sy = crop_data[frame]
                crop_filedict[framei][cam] = tuple(
                    [int(x), int(y), int(w), int(h), float(sx), float(sy)])

    return crop_filedict


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

