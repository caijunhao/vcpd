"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

panda Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of panda robot to pick up a obj.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
from utils import *
from scipy.spatial.transform import Rotation as R
import os
import random
import time
def load_panda(sim, num_envs):
    asset_root = "../../assets"
    panda_asset_file = "urdf/franka_description/robots/panda.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    panda_asset = gym.load_asset(sim, asset_root, panda_asset_file, asset_options)

    # configure panda dofs
    panda_dof_props = gym.get_asset_dof_properties(panda_asset)
    panda_lower_limits = panda_dof_props["lower"]
    panda_upper_limits = panda_dof_props["upper"]
    panda_ranges = panda_upper_limits - panda_lower_limits

    # grippers
    panda_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    panda_dof_props["stiffness"].fill(800.0)
    panda_dof_props["damping"].fill(40.0)

    # default dof states and position targets
    panda_num_dofs = gym.get_asset_dof_count(panda_asset)
    default_dof_pos = np.zeros(panda_num_dofs, dtype=np.float32)
    # grippers open
    default_dof_pos = panda_upper_limits

    default_dof_state = np.zeros(panda_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    default_dof_pos = np.array([default_dof_pos]*num_envs)
    default_dof_pos_tensor = to_torch(default_dof_pos, device=device)
    return default_dof_pos, default_dof_pos_tensor, default_dof_state, panda_asset,  panda_dof_props

def load_obj(sim, asset_path):
    asset_root = '/home/sujc/code/vcpd-master/data/eval/eval_urdf'
    loaded_assets = []
    count = 0

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.disable_gravity = True
    # asset_options.fix_base_link = True
    for asset in asset_path:
        print("Loading asset '%s' from '%s'" % (asset, asset_root))
        current_asset = gym.load_asset(sim, asset_root, asset, asset_options)
        if current_asset is None:
            print("*** Failed to load asset '%s'" % (asset))
            quit()
        loaded_assets.append(current_asset)
        obj_prop = gymapi.RigidShapeProperties()
        obj_prop.friction = 1.0
        obj_prop.restitution = 0.9
        gym.set_asset_rigid_shape_properties(current_asset, [obj_prop])
        count += 1
        if count == num_envs:
            break
    return loaded_assets

def step_update(sim):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)

def get_gripper_dof(num_envs, device, action):
    if action == 'close':
        grip_acts =  torch.Tensor([[0., 0.]] * num_envs).to(device)
    elif action == 'open':
        grip_acts =  torch.Tensor([[0.04, 0.04]] * num_envs).to(device)
    else:
        raise Exception("Gripper action should be string \"close\" or \"open\"")
    return grip_acts

def get_obj_pos(origin_grasp_center, origin_gripper_rot,origin_gripper_pos,
                gripper_pos, gripper_quat,
                depth = 0.10327,new_depth=None):
    origin_obj_rot = np.array([[1,0,0],[0,1,0],[0,0,1]])
    grasp_obj_pos = np.array([0,0,0])
    if torch.is_tensor(gripper_quat):
        gripper_quat = gripper_quat.cpu().numpy()
        gripper_pos = gripper_pos.cpu().numpy()

    if gripper_quat.shape[0] == 4:
        gripper_rot = R.from_quat(gripper_quat).as_matrix()
    else:
        gripper_rot = gripper_quat

    # obj_rot = origin_obj_rot @ np.linalg.inv(origin_gripper_rot) @ gripper_rot
    obj_rot = np.linalg.inv(origin_gripper_rot) @ gripper_rot @ origin_obj_rot

    # compute transform in origin loaded frames where
    # obj: quat[0,0,0,1], pos:[0,0,0]
    # gripper: origin_gripper_rot,origin_gripper_pos
    g2o = np.concatenate([origin_gripper_rot, np.expand_dims(origin_gripper_pos, -1)], axis=-1)
    g2o = np.concatenate([g2o, [[0,0,0,1]]], axis = 0)
    c_in_g = (np.linalg.inv(g2o) @ np.concatenate([origin_grasp_center, [1]]))[:3]
    assert c_in_g[2] - depth < 1e-5,  "grasp info loaded wrong"
    o_in_g = np.linalg.inv(g2o)[:3,3]
    co_in_g = o_in_g - c_in_g
    if new_depth is not None:
        c_in_g = np.array([0,0,new_depth])
        o_in_g = c_in_g + co_in_g

    # compute transform in current frames where
    # gripper: euler:[0, np.pi, 0], pos:[0,0,1]
    g2w = np.concatenate([gripper_rot, np.expand_dims(gripper_pos, -1)], axis=-1)
    g2w = np.concatenate([g2w, [[0,0,0,1]]], axis = 0)
    o_in_w = (g2w @ np.concatenate([o_in_g, [1]]))[:3]
    c_in_w = (g2w @ np.concatenate([c_in_g, [1]]))[:3]
    assert c_in_w[:2].sum() < 1e-6 and gripper_pos[2] - c_in_w[2] - depth < 1e-6, "current obj pos computed wrong"
    obj_pos = o_in_w
    # rotation from origin coordinate to current
    # rot = np.linalg.inv(gripper_rot) @ origin_gripper_rot
    # center_pos_2 = gripper_pos  + rot @ origin_gripper_rot[:,2] * depth
    # origin_center = origin_gripper_rot @ origin_grasp_center
    # g2c = rot @ (origin_grasp_center - origin_gripper_pos)
    # center_pos_ = gripper_pos + g2c
    # c2o = rot @ (grasp_obj_pos - origin_grasp_center)
    # obj_pos = center_pos_ + c2o
    # TODO: check here obj_pos==obj_pos_ , value of x & y should be zero;  center_pos_ ==center_pos_ 2

    obj_quat = R.from_matrix(obj_rot).as_quat()

    invalid_obj_pos = np.copy(gripper_pos)
    invalid_obj_pos[:2] += np.random.rand(2)/2
    invalid_obj_pos[2] = 0.5
    invalid_obj_quat = gripper_quat

    invalid = origin_grasp_center[0] == 100
    final_obj_pos = np.where(invalid, invalid_obj_pos, obj_pos)
    final_obj_quat = np.where(invalid, invalid_obj_quat, obj_quat)
    return final_obj_pos, final_obj_quat

# set random seed
np.random.seed(142)

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=2)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
# Add custom arguments
custom_parameters = [{"name": "--headless", 'type': bool, "default": True, "help": "Direct False, GUI True"},
                     {"name": "--obj_asset_dir", 'type':str,
                      "default":"/home/sujc/code/vcpd-master/data/eval/eval_urdf", "help": "obj model path"},
                     {"name": "--obj_info_path", 'type':str,
                      "default":"/home/sujc/code/vcpd-master/data/eval/eval_grasp_info", "help": "grasp info path"},
                     {"name": "--mode", 'type':str, "default":"eval", "help": "train or eval"}
                     ]
args = gymutil.parse_arguments(
    description="grasping objects using panda grippers",
    custom_parameters=custom_parameters,
)

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'
asset_dir = args.obj_asset_dir
info_path = args.obj_info_path
mode = args.mode

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05


num_assets = 0
success = 0
count = 0
category ='ABCDEFG' if mode == 'eval' else 'ABCDEFGHIJKLMNOPQRSTUVWX'
suc_index = {}
count_index = {}
for i in range(len(category)):
    suc_index[category[i]] = 0
    count_index[category[i]] = 0

for file in os.listdir(asset_dir):
    if not file.find('.urdf') > 0:
        continue
    print('load infos:{}-------------------'.format(file))
    asset_path, asset_name, position, quats,\
    rot, grasp_gripper_positions = asset_preload(asset_dir,num=1,filename=file,info_path=info_path)
    if asset_path is None:
        print('No grasable and collision_free pairs.')
        continue
    # create sim
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise Exception("Failed to create sim")

    # create viewer
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise ValueError('*** Failed to create viewer')

    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, 2 * spacing)

    # configure env grid
    num_envs = np.shape(position)[0]
    num_per_row = int(math.sqrt(num_envs))

    print("File: %s, creating %d environments" %(file[:-5], num_envs))

    # load assets
    default_dof_pos, default_dof_pos_tensor, default_dof_state,\
    panda_asset, panda_dof_props = load_panda(sim, num_envs)
    obj_assets = load_obj(sim, asset_path)

    envs = []
    obj_root_idxs=[]
    obj_handles = []
    panda_handles = []
    hand_root_idxs = []

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0
    gym.add_ground(sim, plane_params)

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add panda
        panda_pose = gymapi.Transform()
        panda_pose.p = gymapi.Vec3(0, 0, 1)
        panda_pose.r = gymapi.Quat.from_euler_zyx(0, math.pi, 0)
        a=R.from_euler('zyx',[np.pi,0,0]).as_matrix()
        gymapi.Quat()
        panda_handle = gym.create_actor(env, panda_asset, panda_pose, "panda_%s"%i, i, 1)
        panda_handles.append(panda_handle)

        gym.set_actor_dof_properties(env, panda_handle, panda_dof_props)
        gym.set_actor_dof_states(env, panda_handle, default_dof_state, gymapi.STATE_ALL)

        # set initial position targets
        gym.set_actor_dof_position_targets(env, panda_handle, default_dof_pos)

        # get inital hand pose
        hand_handle = gym.find_actor_rigid_body_handle(env, panda_handle, "panda_hand")
        # get global root index of obj in rigid body state tensor
        hand_root_idx = gym.get_actor_index(env, hand_handle, gymapi.DOMAIN_SIM)
        hand_root_idxs.append(hand_root_idx)

        # add obj
        panda_pos = np.array([0.0, 0.0, 1.0])
        panda_quat = np.array([0.0, 1.0, 0.0, 0.0])
        origin_grasp_center = position[i]
        origin_gripper_rot = rot[i]
        origin_gripper_pos = grasp_gripper_positions[i]
        init_obj_pos, init_obj_quat = get_obj_pos(origin_grasp_center,
                                                  origin_gripper_rot,
                                                  origin_gripper_pos,
                                                  panda_pos,
                                                  panda_quat,
                                                  depth=0.10327)
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(init_obj_pos[0], init_obj_pos[1], init_obj_pos[2])
        obj_pose.r = gymapi.Quat(init_obj_quat[0], init_obj_quat[1], init_obj_quat[2], init_obj_quat[3])
        obj_handle = gym.create_actor(env, obj_assets[0], obj_pose, asset_name[0], i, 0)
        obj_handles.append(obj_handle)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # get global root index of obj in rigid body state tensor
        obj_root_idx = gym.get_actor_index(env, obj_handle, gymapi.DOMAIN_SIM)
        obj_root_idxs.append(obj_root_idx)

    # point camera at middle env
    cam_pos = gymapi.Vec3(4, 3, 2)
    cam_target = gymapi.Vec3(-4, -3, 0)
    middle_env = envs[num_envs // 2 + num_per_row // 2]
    if not args.headless:
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # ==== prepare tensors =====
    # from now on, we will use the tensor API that can run on CPU or GPU
    gym.prepare_sim(sim)

    # get dof state tensor
    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_pos = dof_states[:, 0].view(num_envs, 2, 1)
    dof_vel = dof_states[:, 1].view(num_envs, 2, 1)

    # Set action tensors
    pos_action = torch.zeros_like(dof_pos).squeeze(-1)
    effort_action = torch.zeros_like(pos_action)

    # get root body tensor
    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_tensor = gymtorch.wrap_tensor(_root_tensor)


    event = 0
    c = close = 120
    s = shake = 10
    w = wait = 20
    init_obj_pos = None
    while True:

        step_update(sim)

        # get real-time poses of actors
        obj_pos = root_tensor[obj_root_idxs,:3]
        hand_pos = root_tensor[hand_root_idxs,:3]
        hand_quat = root_tensor[hand_root_idxs,3:7]
        '''
        event= :
        1 : get init obj pos
        w~w+c : close gripper
        w+c : enable gravity of objects
        w+c~w+c+4*s : move gripper up and down
        w+c+4*s~w+c+8*s : shake gripper along y axis
        w+c+8*s : record success
        w+c+8*s~w+2*c+8*s : open gripper
        '''
        # Note: delete up and down here
        if event == 1:
            init_obj_pos = obj_pos

        if w <= event < w + c:
            pos_action[:] = get_gripper_dof(num_envs, device, action='close')
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

        elif event == w + c:
            for i in range(num_envs):
                properties = gym.get_actor_rigid_body_properties(envs[i], obj_handles[i])[0]
                properties.flags = 0
                gym.set_actor_rigid_body_properties(envs[i], obj_handles[i], [properties])

        # elif w + c + 1 <= event < w + c + 4 * s:
        #     if event < w + c + s or w + c + 2 * s <= event < w + c + 3 * s:
        #         height = 0.01
        #     else:
        #         height = -0.01
        #     rigid_obj = torch.clone(root_tensor)
        #     rigid_obj[hand_root_idxs, 2] += height
        #     actor_indices = torch.tensor([hand_root_idxs], dtype=torch.int32, device=device)
        #     gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
        #                                             gymtorch.unwrap_tensor(actor_indices),
        #                                             num_envs)
        # elif w + c + 4 * s <= event < w + c + 8 * s:
        #     if event < w + c + 5 * s or w + c + 6 * s <= event < w + c + 7 * s:
        #         angle = 2
        #     else:
        #         angle = -2
        elif w + c + 1 <= event < w + c + 4 * s:
            if event < w + c + s or w + c + 2 * s <= event < w + c + 3 * s:
                angle = 2
            else:
                angle = -2
            rigid_obj = torch.clone(root_tensor)
            hand_quat = rigid_obj[hand_root_idxs, 3:7].cpu().numpy()

            hand_euler = R.from_quat(hand_quat).as_euler('zyx',degrees=True)
            hand_euler[:, 1] += angle
            hand_quat = R.from_euler('zyx',hand_euler,degrees=True).as_quat()

            hand_quat = torch.from_numpy(hand_quat).to(device)
            rigid_obj[hand_root_idxs, 3:7] = hand_quat.to(torch.float32)
            actor_indices = torch.tensor([hand_root_idxs], dtype=torch.int32, device=device)
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                    gymtorch.unwrap_tensor(actor_indices),
                                                    num_envs)
        elif event == w + c + 4 * s:
            su = obj_pos[:, 2] > init_obj_pos[:, 2] - 0.05
            su = np.int0(torch.sum(su).cpu().numpy())
            success += su
            count += num_envs
            key = file[0]
            suc_index[key] += su
            count_index[key] += num_envs
            print('{}, {} grasps, {}success---------------'.format(file, num_envs, su))

        elif w + c + 4 * s < event < w + 2 * c + 4 * s:
            pos_action[:] = get_gripper_dof(num_envs, device, action='open')
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))


        elif event > w + 2 * c + 4 * s:
            break

        event += 1
        gym.step_graphics(sim)
        if not args.headless:
            # render the viewer
            gym.draw_viewer(viewer, sim, True)

            # Wait for dt to elapse in real time to sync viewer with
            # simulation rate. Not necessary in headless.
            gym.sync_frame_time(sim)

            # Check for exit condition - user closed the viewer window
            if gym.query_viewer_has_closed(viewer):
                break
    # cleanup
    print('{}, {} grasps, {}success, {:.2f}%---------------'.format(file, count, success, 100*success/count))
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
for i in range(len(category)):
    key = category[i]
    print("{}:, {} grasps, {} success, {:.2f}%".format(key,
                                                   count_index[key],
                                                   suc_index[key],
                                                   100*suc_index[key]/count_index[key]))
