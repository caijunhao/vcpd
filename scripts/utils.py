import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch

def get_grasp_pose(obj_name,
                   data_path=None
                   ):
    if data_path is None:
        data_path = '/home/sujc/code/vcpd-master/dataset/grasp_info/train_grasp_info'
    with open(os.path.join(data_path, '{}_info.json'.format(obj_name)), 'r', encoding='utf8') as f:
        json_data = json.load(f)
    if json_data['num_col-free_poses'] == 0:
        return None, None, None, None
    positions = []
    quats = []
    rots = []
    gripper_positions = []
    centers = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'centers')))
    widths = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'widths')))
    collisions = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'collisions')))
    quaternions = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'quaternions')))
    scores = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'antipodal_mean')))
    for i in range(centers.shape[0]):
        grasp_ids = np.where(collisions[i] == 0)[0]
        if grasp_ids.shape[0] == 0:
            continue
        center = centers[i]
        width = widths[i]
        quaternion = quaternions[i]
        score = scores[i]
        if score < 0.99:
            continue
        # if len(positions)>500:
        #     break
        for j in range(grasp_ids.shape[0]):
            grasp_id = grasp_ids[j]
            quat = quaternion[grasp_id]
            rot = R.from_quat(quat).as_matrix()
            pos = center.copy()
            gripper_pos = center.copy() - rot[:, 2] * 0.10327

            positions.append(pos)
            quats.append(quat)
            rots.append(rot)
            gripper_positions.append(gripper_pos)
    return positions, quats, rots, gripper_positions

def asset_preload(dir, num, filename=None, info_path=None):
    asset_paths = []
    asset_names = []
    positions = []
    quats = []
    rots = []
    gripper_positions = []
    count = 0
    n_max = 0

    for files in os.listdir(dir):
        if filename is not None:
            file = os.path.join(dir, filename)
        else:
            file = os.path.join(dir, files)

        if os.path.isfile(file) and file.find('.urdf') > 0:
            i = file.rfind('/')
            asset_paths.append(file[i+1:])

            name = file[i+1: -5]
            asset_names.append(name)

            position, quat, rot, gripper_pos = get_grasp_pose(name,info_path)
            if position is None:
                return None, None, None, None, None, None
            if len(position) > n_max:
                n_max = len(position)
            positions.append(position)
            quats.append(quat)
            rots.append(rot)
            gripper_positions.append(gripper_pos)
            count += 1
            if count == num:
                break
        if filename is not None:
            break
    for i in range(count):
        # n = n_max - len(positions[i])
        # for j in range(n):
        #     positions[i].append(np.array([100] * 3))
        #     gripper_positions[i].append(np.array([100] * 3))
        #     quats[i].append(np.array([100] * 4))
        #     rots[i].append( 100 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) )
        positions[i] = np.array(positions[i])
        quats[i] = np.array(quats[i])
        rots[i] = np.array(rots[i])
        gripper_positions[i] = np.array(gripper_positions[i])
    try:
        positions = np.squeeze(np.array(positions).transpose((1,0,2)))
        quats = np.squeeze(np.array(quats).transpose((1,0,2)))
        rots = np.squeeze(np.array(rots).transpose((1,0,2,3)))
        gripper_positions = np.squeeze(np.array(gripper_positions).transpose((1,0,2)))
        if len(positions.shape) == 1:
            positions = np.expand_dims(positions,axis=0)
            quats = np.expand_dims(quats, axis=0)
            rots = np.expand_dims(rots, axis=0)
            gripper_positions = np.expand_dims(gripper_positions, axis=0)
    except:
        print('error')
        print(filename)
    return asset_paths, asset_names, positions, quats, rots, gripper_positions

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(dpose, device="cuda:0"):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u

def control_osc(dpose, device="cuda:0"):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


def get_obj_file(path):
    '''
    example:
    file -> obj_name: 'A00_0.urdf' -> A00_0; 'C2.urdf' -> C2
    '''
    obj_files = []
    file= open(path,"r")
    for line in file.readlines():
        line = line.strip('\n')
        obj_files.append(line)
    file.close()
    return obj_files