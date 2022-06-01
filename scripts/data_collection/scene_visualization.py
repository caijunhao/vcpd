from sim.utils import get_multi_body_template, add_sphere
import pybullet as p
import pybullet_data
import numpy as np
import argparse
import json
import os


def main(args):
    p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.37, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0, 0, 0])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # plane_id = p.loadURDF('plane.urdf')
    folders = os.listdir(args.path)
    np.random.shuffle(folders)
    for folder in folders:
        path = os.path.join(args.path, folder)
        with open(os.path.join(path, folder+'_sdf_info.json'), 'r') as f:
            sdf_info = json.load(f)
            vl = sdf_info['voxel_length']
            o = np.asarray(sdf_info['origin'], dtype=np.float32)
        vis_params, col_params, body_params = get_multi_body_template()
        vis_params['fileName'] = os.path.join(path, folder + '_mesh.obj')
        col_params['fileName'] = os.path.join(path, folder + '_mesh.obj')
        m = p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(**col_params),
                              baseVisualShapeIndex=p.createVisualShape(**vis_params),
                              **body_params)
        p.changeVisualShape(m, -1, rgbaColor=[1, 1, 1, 0.7])
        ids = set([file.split('_')[0] for file in os.listdir(path) if 'contact' in file])
        for idx in ids:
            pos_cp1 = np.load(os.path.join(path, idx+'_pos_contact1.npy'))
            pos_cp2 = np.load(os.path.join(path, idx+'_pos_contact2.npy'))
            neg_cp1 = np.load(os.path.join(path, idx + '_neg_contact1.npy'))
            neg_cp2 = np.load(os.path.join(path, idx + '_neg_contact2.npy'))
            num_pos, num_neg = pos_cp1.shape[0], neg_cp1.shape[0]
            pos_ids = np.random.choice(np.arange(num_pos), size=min(num_pos, 777), replace=False)
            neg_ids = np.random.choice(np.arange(num_neg), size=min(num_neg, 777), replace=False)
            pos_cp1, pos_cp2 = pos_cp1[pos_ids], pos_cp2[pos_ids]
            neg_cp1, neg_cp2 = neg_cp1[neg_ids], neg_cp2[neg_ids]
            pos_cp1, pos_cp2, neg_cp1, neg_cp2 = pos_cp1*vl+o, pos_cp2*vl+o, neg_cp1*vl+o, neg_cp2*vl+o
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            cp1 = [add_sphere(cp, rgb=[1, 0, 0]) for cp in pos_cp1]
            cp2 = [add_sphere(cp, rgb=[0, 1, 0]) for cp in pos_cp2]
            lines = [p.addUserDebugLine(cp1, cp2, lineColorRGB=np.random.uniform(size=3)) for cp1, cp2 in
                     zip(pos_cp1, pos_cp2)]
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            [p.removeBody(cp) for cp in cp1]
            [p.removeBody(cp) for cp in cp2]
            [p.removeUserDebugItem(line) for line in lines]
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            cp1 = [add_sphere(cp, rgb=[1, 0, 0]) for cp in neg_cp1]
            cp2 = [add_sphere(cp, rgb=[0, 1, 0]) for cp in neg_cp2]
            lines = [p.addUserDebugLine(cp1, cp2, lineColorRGB=np.random.uniform(size=3)) for cp1, cp2 in
                     zip(neg_cp1, neg_cp2)]
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            [p.removeBody(cp) for cp in cp1]
            [p.removeBody(cp) for cp in cp2]
            [p.removeUserDebugItem(line) for line in lines]
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.removeBody(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stacked scene construction.')
    parser.add_argument('--path',
                        type=str,
                        required=True,
                        help='path to the save the scene data')
    main(parser.parse_args())
