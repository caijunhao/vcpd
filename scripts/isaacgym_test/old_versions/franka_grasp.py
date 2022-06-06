import argparse

parser = argparse.ArgumentParser(description='Stacked scene construction.')
parser.add_argument('--isaacgym_path',
                    default='/home/sujc/code/isaacgym/python',
                    help='path to installed isaac gym(should be like .../isaacgym/python)')
parser.add_argument('--config',
                    type=str,
                    default='/home/sujc/code/vcpd/config/config.json',
                    help='path to the config file.')
arg = parser.parse_args()

import json

with open(arg.config, 'r') as config_file:
    cfg = json.load(config_file)

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys

sys.path.append(arg.isaacgym_path)
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import matplotlib.pyplot as plt
from sdf import SDF
from sim.camera import Camera
from isaac.utils import get_tray, add_noise, PandaGripper
from cpn.utils import sample_contact_points, select_gripper_pose
from cpn.model import CPN

import time as ti

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(dpose, device="cuda:0", damping = 0.05):
    global j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u

def get_gripper_dof(num_envs, action, device):
    if action == 'close':
        grip_acts = torch.Tensor([[0., 0.]] * num_envs).to(device)
    elif action == 'open':
        grip_acts = torch.Tensor([[0.04, 0.04]] * num_envs).to(device)
    else:
        raise Exception("Gripper action should be string \"close\" or \"open\"")
    return grip_acts

def load_obj(gym, sim, asset_root, asset_path, asset_options=None):
    loaded_assets = []
    count = 0
    if asset_options is None:
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.override_com = True
        asset_options.override_inertia = True
    for asset in asset_path:
        print("Loading asset '%s' from '%s'" % (asset, asset_root))
        current_asset = gym.load_asset(sim, asset_root, asset, asset_options)
        if current_asset is None:
            print("*** Failed to load asset '%s'" % (asset))
            quit()
        obj_prop = gymapi.RigidShapeProperties()
        obj_prop.friction = 3.0
        obj_prop.restitution = 0.9
        obj_prop.rolling_friction = 3.0
        gym.set_asset_rigid_shape_properties(current_asset, [obj_prop])
        loaded_assets.append(current_asset)
        count += 1
        if count == args.num_objects:
            break
    return loaded_assets

def create_obj_assets(type=1):
    if type=='primitive' or type==0:
        asset_root = '/home/sujc/code/vcpd/data/primitive/primitive_urdf'
        paths = os.listdir(asset_root)
        asset_paths = []
        for path in paths:
            asset_paths.append(path + '/' + path + '.urdf')
        obj_assets = load_obj(gym, sim, asset_root, asset_paths)

    # simple objects
    elif type=='simple' or type==1:
        box_size = 0.045
        asset_options = gymapi.AssetOptions()
        obj_assets = []
        for i in range(args.num_objects):
            obj_assets.append(gym.create_box(sim, box_size, box_size, box_size, asset_options))
            obj_prop = gymapi.RigidShapeProperties()
            obj_prop.friction = 300.0
            obj_prop.restitution = 0.9
            obj_prop.rolling_friction = 300.0
            gym.set_asset_rigid_shape_properties(obj_assets[-1], [obj_prop])

    # egad objects
    elif type == 'egad' or type==2:
        asset_root = '/home/sujc/code/vcpd-master/data/train/train_urdf'
        asset_paths = []
        paths = os.listdir(asset_root)
        for path in paths:
            if '.urdf' in path:
                asset_paths.append(path)
        obj_assets = load_obj(gym, sim, asset_root, asset_paths)
    return obj_assets

def sample_a_pose_from_a_sphere(r, target_position,
                                theta_min=0, theta_max=45,
                                phi_min=0, phi_max=360):
    d2r = np.deg2rad
    theta = np.random.uniform(theta_min, theta_max)
    phi = np.random.uniform(phi_min, phi_max)
    theta = d2r(theta)
    phi = d2r(phi)
    shift = np.array([r * np.sin(theta) * np.cos(phi),
                      r * np.sin(theta) * np.sin(phi),
                      r * np.cos(theta)])
    cam_pos = target_position + shift
    z = np.array([r * np.cos(theta) * np.cos(phi),
                  r * np.cos(theta) * np.sin(phi),
                  -r * np.sin(theta)])  # correspond to y axis
    z = z / np.linalg.norm(z)
    x = -shift / np.linalg.norm(shift)
    y = np.cross(z, x)
    cam_rot = np.stack([x, y, z], axis=1)
    T = np.concatenate((cam_rot, cam_pos.reshape((3, 1))), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    return cam_pos, cam_rot, T


def get_cam_pose(env, cam_handle):
    camera_pose = gym.get_camera_transform(sim, env, cam_handle)
    q = camera_pose.r
    p = camera_pose.p
    # ca_pose = gym_pose_to_matrix(
    #     {"r": [q.x, q.y, q.z, q.w], "p": [p.x, p.y, p.z]}
    # )
    r = R.from_quat(np.array([q.x, q.y, q.z, q.w])).as_matrix()
    p = np.array([p.x, p.y, p.z]).reshape((3, 1))

    pose = np.concatenate((r, p), axis=1)
    pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)
    return r, pose

def load_franka(sim, franka_asset_root=None, franka_asset_file=None):
    if franka_asset_root == None:
        franka_asset_root = '/home/sujc/code/isaacgym/assets'
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    franka_asset = gym.load_asset(sim, franka_asset_root, franka_asset_file, asset_options)

    # configure franka dofs
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)

    # use position drive for all dofs
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["damping"][:7].fill(40.0)

    # grippers
    franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][7:].fill(800.0)
    franka_dof_props["damping"][7:].fill(40.0)

    # default dof states and position targets
    franka_num_dofs = gym.get_asset_dof_count(franka_asset)
    default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
    default_dof_pos[:7] = franka_mids[:7]
    # grippers open
    default_dof_pos[7:] = franka_upper_limits[7:]

    default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    # send to torch
    default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

    # get link index of panda hand, which we will use as end effector
    franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
    franka_hand_index = franka_link_dict["panda_hand"]
    return franka_asset, franka_dof_props, default_dof_state, default_dof_pos, franka_hand_index, default_dof_pos_tensor

def board_pose(shift=0, box_z = 1.5):
    poses = []
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.3, box_z / 2) + shift
    pose.r = gymapi.Quat(0, 0, 0, 1)
    poses.append(pose)

    # pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, -0.3, box_z / 2) + shift
    pose.r = gymapi.Quat(0, 0, 0, 1)
    poses.append(pose)

    # pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.3, 0.0, box_z / 2) + shift
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0.5 * math.pi)
    poses.append(pose)

    # pose = gymapi.Transform()
    pose.p = gymapi.Vec3(-0.3, 0.0, box_z / 2) + shift
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0.5 * math.pi)
    poses.append(pose)

    # pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, box_z) + shift
    pose.r = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    poses.append(pose)
    return poses


# initialize gym
gym = gymapi.acquire_gym()
custom_parameters = [{"name": "--headless", 'type': bool, "default": True, "help": "Direct False, GUI True"},
                     {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
                     {"name": "--num_objects", "type": int, "default": 20, "help": "Number of objects in the bin"},
                     {"name": "--obj_asset_dir", 'type': str,
                      "default": "/home/sujc/code/vcpd-master/data/train/train_urdf"},
                     {"name": "--device", 'type': int, "default": 1, "help": "gpu device"},
                     {"name": "--model_path", 'type': str, "default": '/home/sujc/code/vcpd/log/0/cpn_cjh.pth',
                      'help': "cpn model path"}
                     ]
# Add custom arguments
# parse arguments
args = gymutil.parse_arguments(description="Body Physics Properties Example",
                               custom_parameters=custom_parameters, )

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

# ----------------------------------------------------------------
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# set up the env grid

spacing = 2.0
env_lower = gymapi.Vec3(-spacing, -spacing, -0.2)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
device = 'cuda:0'

# create table asset-------------------------------------------------------
table_dims = gymapi.Vec3(0.6, 1.0, 0.02)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create bin asset-------------------------------------------------------------
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
num_objects = args.num_objects
asset_root = "/home/sujc/code/isaacgym/assets"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
shift_x = 0.6
shift = gymapi.Vec3(shift_x, 0.0, table_dims.z)
size_list, pose_list = get_tray(shift=shift)
bins = []
for i in range(len(size_list)):
    bins.append(gym.create_box(sim, size_list[i][0], size_list[i][1], size_list[i][2], asset_options))

# create static box asset----------------------------------------------------------
asset_options.density = 1000000
box_z = 1.5
asset_board = gym.create_box(sim, 0.6, 0.01, box_z, asset_options)
asset_cam_body = gym.create_box(sim, 0.0001, 0.0001, 0.0001, asset_options)


# create object assets -----------------------------------------------------------
obj_assets = create_obj_assets()
# load franka robot
franka_asset, franka_dof_props, default_dof_state, \
default_dof_pos, franka_hand_index, default_dof_pos_tensor = load_franka(sim)

# create list to mantain environment and asset handles------------------------------------------
envs = []
tray_handles = []
obj_handles = []
obj_root_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []
box_handles = []
box_root_idxs = []
cam_body_handles = []
cam_body_root_idxs = []
cams = []
cam_root_idxs = []
depth_tensors = []
color_tensors = []


# set cam props
get_intrinsic = False
cam_z = 0.55
cam_props = gymapi.CameraProperties()
cam_props.width = 640
cam_props.height = 480
cam_props.enable_tensors = True
cam_props.use_collision_geometry = True
print('Creating %d environments' % num_envs)

# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.5)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom_1 = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=(1, 0, 0))
sphere_geom_2 = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=(0, 1, 0))

sphere_geom_1s = gymutil.WireframeSphereGeometry(0.006, 12, 12, sphere_pose, color=(1, 1, 0))
sphere_geom_2s = gymutil.WireframeSphereGeometry(0.006, 12, 12, sphere_pose, color=(0, 1, 1))

# some fixed poses
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(shift_x, 0.0, 0.5 * table_dims.z)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

shift = gymapi.Vec3(shift_x, 0.0, table_dims.z)
board_poses = board_pose(shift=shift)

cam_body_pose = gymapi.Transform()
cam_body_pose.p = gymapi.Vec3(1e-3, 0.0, cam_z) + shift
cam_body_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(90))

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # create bin
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    for ii in range(len(size_list)):
        tray_handles.append(gym.create_actor(env, bins[ii], pose_list[ii], "bin" + str(ii), i, 0))
        gym.set_rigid_body_color(env, tray_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # obj_list = np.random.choice(obj_assets, args.num_objects, replace=False)

    # create 5 boxes
    for j in range(5):
        box_handle = gym.create_actor(env, asset_board, board_poses[j], "board_%s"%j, i, 0)
        box_handles.append(box_handle)
        box_root_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
        box_root_idxs.append(box_root_idx)


    cam_body_handle = gym.create_actor(env, asset_cam_body, cam_body_pose, str(i) + "cam_body", i, 3)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, cam_body_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    cam_body_handles.append(cam_body_handle)
    cam_body_root_idx = gym.get_actor_index(env, cam_body_handle, gymapi.DOMAIN_SIM)
    cam_body_root_idxs.append(cam_body_root_idx)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)


    # create objects
    for j in range(args.num_objects):
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(np.random.uniform(-0.15, 0.15), np.random.uniform(-0.15, 0.15),
                                 np.random.uniform(0.05, 0.5)) + shift
        obj_pose.r = gymapi.Quat.from_euler_zyx(np.random.rand() * math.pi,
                                                np.random.rand() * math.pi,
                                                np.random.rand() * math.pi)
        obj_handle = gym.create_actor(env, obj_assets[j], obj_pose, str(i) + str(j), i, 0)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        obj_handles.append(obj_handle)
        # get global root index of obj in rigid body state tensor
        obj_root_idx = gym.get_actor_index(env, obj_handle, gymapi.DOMAIN_SIM)
        obj_root_idxs.append(obj_root_idx)

    # add camera

    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.attach_camera_to_body(cam_handle, env, cam_body_handle,
                              gymapi.Transform(),
                              gymapi.FOLLOW_TRANSFORM)

    # gym.set_camera_location(cam_handle, env, gymapi.Vec3(1e-3, 0, 0.55), gymapi.Vec3(0, 0, 0))
    # gym.set_camera_transform(cam_handle, env, gymapi.Transform(gymapi.Vec3(1e-3, 0, 0.55), camera_rotation))
    cams.append(cam_handle)
    if not get_intrinsic:
        projection_matrix = np.matrix(gym.get_camera_proj_matrix(sim, env, cam_handle))
        fx = 1 / (2 / projection_matrix[0, 0] / cam_props.width)
        fy = 1 / (2 / projection_matrix[1, 1] / cam_props.height)
        cx = cam_props.width / 2
        cy = cam_props.height / 2
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        get_intrinsic = True
        # view_matrix = np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle))
    # obtain camera tensor
    depth_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    color_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    print("Got camera tensor with shape", depth_tensor.shape)

    # wrap camera tensor in a pytorch tensor
    torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor)
    depth_tensors.append(torch_depth_tensor)
    torch_color_tensor = gymtorch.wrap_tensor(color_tensor)
    color_tensors.append(torch_color_tensor)

# look at the first env
cam_pos = gymapi.Vec3(0.5, 0, 0.5)
cam_target = gymapi.Vec3(0, 0, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# ==== prepare tensors =====
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)

# get root body tensor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)

# analaysis model-------------------------------------------------------
cpn = CPN()
cpn.load_network_state_dict(device=device, pth_file=args.model_path)
cpn.to(device)
vol_bnd = np.array([cfg['sdf']['x_min'], cfg['sdf']['y_min'], cfg['sdf']['z_min'],
                    cfg['sdf']['x_max'], cfg['sdf']['y_max'], cfg['sdf']['z_max']]).reshape(2, 3) + \
          np.array([[shift.x, shift.y, shift.z]])

voxel_length = cfg['sdf']['resolution']
tsdf = SDF(vol_bnd, voxel_length, rgb=False, device=device)
pg = PandaGripper('../../assets')

# paraemeters
grasp_poses = []
grasps_widths = []
grasp_rots = []
grasp_quats = []
depths = []
cam_poses = []
t = 0
init_d = 0.5
delta_d = 0.005
s = stable = 150
f = cfg['test']['frame']
o = down = 180
a = away = 100
c = close = 80
isaac_rot = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
panda_rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).astype(np.float32)

# osc parameters
p1 = gymapi.Transform()
p2 = gymapi.Transform()

controller = 'ik'
time = {'stable': s,
        'tsdf': s + f,
        'down': s + f + o,
        'close': s + f + o + c,
        'up': s + f + o + c + a,
        'open': s + f + o + 2 * c + a,
}
succ = 0
count = 0
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    if t == 0:
        init_pos_action = dof_pos.squeeze(-1).clone()
        init_hand_pos = hand_pos.clone()
        init_hand_rot = hand_rot.clone()
        pos_action = init_pos_action.clone()

    if t == time['stable'] - 1:
        rigid_obj = torch.clone(root_tensor)
        rigid_obj[box_root_idxs, :2] = -1.5
        rigid_obj[box_root_idxs, 2] = 0.01 / 2
        rigid_obj[box_root_idxs, 3:7] = 0
        rigid_obj[box_root_idxs, 3] = rigid_obj[box_root_idxs, 6] = 0.70107
        actor_indices = torch.tensor([box_root_idxs], dtype=torch.int32, device=device)
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                gymtorch.unwrap_tensor(actor_indices),
                                                len(box_root_idxs))

    if t == time['stable']:
        init_rot, w2c = get_cam_pose(envs[-1], cams[-1])
    # update the viewer
    gym.step_graphics(sim)

    # render sensors and refresh camera tensors
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    if time['stable'] < t <= time['tsdf']:
        depth_s = []
        for i in range(num_envs):
            depth = -torch.clone(depth_tensors[i]).cpu().numpy()
            depth[depth == np.inf] = 0
            depth = add_noise(depth, intrinsic)
            depth_s.append(depth)
        depths.append(depth_s)
        cam_poses.append(w2c)

        cam_pos, cam_rot, w2c = sample_a_pose_from_a_sphere(0.55, np.array([shift.x, shift.y, shift.z]))
        pos = torch.from_numpy(cam_pos).to(device)
        quat = R.from_matrix(cam_rot).as_quat()
        quat = torch.from_numpy(quat).to(device)

        rigid_obj = torch.clone(root_tensor)
        rigid_obj[cam_body_root_idxs, 0:3] = pos.to(torch.float32)
        rigid_obj[cam_body_root_idxs, 3:7] = quat.to(torch.float32)

        actor_indices = torch.tensor([cam_body_root_idxs], dtype=torch.int32, device=device)
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                gymtorch.unwrap_tensor(actor_indices),
                                                len(cam_body_root_idxs))


    if t == time['tsdf']:

        for i in range(num_envs):
            for j in range(f):
                tsdf.integrate(depths[j][i], intrinsic, cam_poses[j], isaac_rot=isaac_rot)
            depth_heightmap = tsdf.get_heightmap()
            # plt.imshow(depth_heightmap.cpu().numpy())
            # plt.show()
            # plt.imshow(depths[0][0])
            # plt.show()
            tsdf.write_mesh('out.ply', *tsdf.compute_mesh(step_size=3))
            cp1, cp2 = sample_contact_points(tsdf)
            ids_cp1, ids_cp2 = tsdf.get_ids(cp1), tsdf.get_ids(cp2)
            sample = dict()
            sample['sdf_volume'] = tsdf.gaussian_blur(tsdf.post_processed_volume).unsqueeze(dim=0).unsqueeze(dim=0)
            sample['ids_cp1'] = ids_cp1.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
            sample['ids_cp2'] = ids_cp2.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
            out = torch.squeeze(cpn.forward(sample))
            pos, rot, width, cp1, cp2, cp1s, cp2s = select_gripper_pose(tsdf, pg.vertex_sets, out, cp1, cp2,
                                                                        cfg['gripper']['depth'])
            if cp1[0] > cp2[0]:
                rot = np.dot(rot, panda_rot)
                tmp = cp1
                cp1 = cp2
                cp2 = tmp
            # visualize
            color = gymapi.Vec3(1, 0, 0)
            p1.p = gymapi.Vec3(cp1[0], cp1[1], cp1[2])
            p2.p = gymapi.Vec3(cp2[0], cp2[1], cp2[2])
            gymutil.draw_line(p1.p, p2.p, color, gym, viewer, envs[i])
            gymutil.draw_lines(sphere_geom_1, gym, viewer, envs[i], p1)
            gymutil.draw_lines(sphere_geom_2, gym, viewer, envs[i], p2)

            grasp_poses.append(pos)
            grasp_rots.append(rot)
            grasp_quats.append(R.from_matrix(rot).as_quat())
            grasps_widths.append(width)
            tsdf.reset()

        grasp_poses = np.array(grasp_poses)
        grasp_rots = np.array(grasp_rots)
        grasps_widths = np.array(grasps_widths)
        grasp_quats = np.array(grasp_quats)
        grasp_quats = torch.from_numpy(grasp_quats).to(device)



        gp = gymapi.Transform()
        gp.p = gymapi.Vec3(grasp_poses[0, 0], grasp_poses[0, 1], grasp_poses[0, 2])
        gp.r = gymapi.Quat(grasp_quats[0, 0], grasp_quats[0, 1], grasp_quats[0, 2], grasp_quats[0, 3])
        gymutil.draw_lines(axes_geom, gym, viewer, envs[0], gp)
        for j in range(cp1s.shape[0]):
            cp1 = cp1s[j]
            cp2 = cp2s[j]
            color = gymapi.Vec3(1, 0, 0)
            p1.p = gymapi.Vec3(cp1[0], cp1[1], cp1[2])
            p2.p = gymapi.Vec3(cp2[0], cp2[1], cp2[2])
            gymutil.draw_line(p1.p, p2.p, color, gym, viewer, envs[0])
            gymutil.draw_lines(sphere_geom_1s, gym, viewer, envs[0], p1)
            gymutil.draw_lines(sphere_geom_2s, gym, viewer, envs[0], p2)

    if time['tsdf'] < t <= time['down']:
        if t <= time['down']/2:
            goal_pos = grasp_poses - grasp_rots[..., 2] * 0.06
            goal_pos = torch.from_numpy(goal_pos).to(device)
        else:
            goal_pos = torch.from_numpy(grasp_poses).to(device)
        goal_rot = grasp_quats.to(torch.float32)

        if t == time['down']:
            print('orn err:%s, pos_err:%s'%((goal_rot - hand_rot).sum(dim=-1), (goal_pos-hand_pos).sum(dim=-1)))
        pos_err = (goal_pos - hand_pos)*0.16
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # Deploy control

        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    if t == time['down']:
        pos_action[:, 7:9] = get_gripper_dof(num_envs, action='close', device=device)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    if t == time['close']:
        pos_action[...] = init_pos_action[...]
        pos_action[:, 7:9] = get_gripper_dof(num_envs, action='close', device=device)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    if t == time['up']:
        count += 1
        succ += (dof_pos.squeeze(-1).clone()[:, 7] > 0.008).cpu().numpy().sum()
        print('succ:%s, grasps:%s, rate:%.2f%%'%(succ,count,succ/count*100))
        pos_action[:, 7:9] = get_gripper_dof(num_envs, action='open', device=device)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))

    if t > time['open']:
        gym.clear_lines(viewer)
        t = time['stable']
        grasp_poses = []
        grasps_widths = []
        grasp_rots = []
        grasp_quats = []
        depths = []
        cam_poses = []
    gym.end_access_image_tensors(sim)


    gym.draw_viewer(viewer, sim, True)
    t += 1
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
