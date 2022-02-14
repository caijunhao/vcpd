from trimesh.proximity import ProximityQuery
from utils import *
from sdf.sdf import TSDF
import imageio as io
import numpy as np
import trimesh
import argparse
import shutil
import torch
import json
import time
import os

torch.set_default_dtype(torch.float32)
tf32 = torch.float32
nf32 = np.float32


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    with open(args.config, 'r') as config_file:
        cfg = json.load(config_file)['sdf']
    vol_bnd = np.array([cfg['x_min'], cfg['x_max'],
                        cfg['y_min'], cfg['y_max'],
                        cfg['z_min'], cfg['z_max']], dtype=nf32).reshape(3, 2)
    res = cfg['resolution']
    tsdf = TSDF(vol_bnd, res, rgb=False)
    vol_bnd = tsdf.vol_bnd
    num_frame = cfg['save_volume'][-1] + 1
    for folder in sorted(os.listdir(args.scene_dir)):
        print(folder)
        total = 0
        scene_dir = os.path.join(args.scene_dir, folder)
        sdf_dir = os.path.join(args.sdf_dir, folder)
        if not os.path.exists(sdf_dir):
            os.makedirs(sdf_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tsdf volume with contact points info.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to the config file.')
    parser.add_argument('--mesh_dir',
                        type=str,
                        required=True,
                        help='path to the mesh file')
    parser.add_argument('--scene_dir',
                        type=str,
                        required=True,
                        help='path to the scene data')
    parser.add_argument('--sdf_dir',
                        type=str,
                        required=True,
                        help='path to the sdf data')
    parser.add_argument('--cuda_device',
                        default='0',
                        type=str,
                        help='id of nvidia device.')
    main(parser.parse_args())
