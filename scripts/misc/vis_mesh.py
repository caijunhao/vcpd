from sim.utils import *
from sim.objects import RigidObject
import xml.etree.ElementTree as et
import pybullet as p
import numpy as np
import argparse
import os


def main(args):
    # PyBullet initialization
    p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # scene initialization
    mesh_list = os.listdir(args.mesh)
    mesh_list = np.sort(mesh_list)
    num_mesh = len(mesh_list)
    n = np.ceil(np.sqrt(num_mesh))
    length = 0.1
    obj_list = list()
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    for idx in range(num_mesh):
        mesh_name = mesh_list[idx]
        i, j = idx // n, idx % n
        x, y = (i - n / 2) * length, (j - n / 2) * length
        pos, quat = np.array([x, y, 0]), np.array([0, 0, 0, 1])
        urdf_path = os.path.join(args.mesh, mesh_name, mesh_name + '.urdf')
        scale = [elem for elem in et.parse(urdf_path).getroot().iter('mesh')][0].attrib['scale'].split(' ')
        scale = [float(s) for s in scale]
        vis_params, col_params, body_params = get_multi_body_template()
        vis_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_vis.obj')
        vis_params['meshScale'] = scale
        col_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_col.obj')
        col_params['meshScale'] = scale
        body_params['basePosition'], body_params['baseOrientation'] = pos, quat
        obj_list.append(RigidObject(mesh_name,
                                    vis_params=vis_params,
                                    col_params=col_params,
                                    body_params=body_params))
        obj_list[-1].change_color()
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.57, cameraYaw=90, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
    [p.removeBody(o.obj_id) for o in obj_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stacked scene construction.')
    parser.add_argument('--mesh',
                        type=str,
                        required=True,
                        help='path to the mesh set')
    main(parser.parse_args())
