from isaacgym import gymutil
import time as tt
aa=tt.time()
custom_parameters = [{"name": "--headless", 'type': int, "default": 0, "help": "Direct 1, GUI 0"},
                     {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create"},
                     {"name": "--num_objects", "type": int, "default": 5, "help": "Number of objects in the bin"},
                     {"name": "--gpu", "type": int, "default": 1, "help": "gpu device"},
                     {"name": "--grasp_per_env", "type": int, "default": 1, "help": "grasps attempts per env"},
                     {"name": "--obj_type", "type": int, "default": 1, "help": "0:primitive; 1:random; 2:kit"},
                     {"name": "--rn_used", "type": int, "default": 0, "help": "0:vpn 1:vpn + rn"},
                     {"name": "--vpn_path", "type": str, "default": "../../log/vpn.pth",
                      "help": "vpn model path"},
                     {"name": "--rn_path", "type": str, "default": "../../log/rn.pth",
                      "help": "rn model path"},
                     {"name": "--config", "type": str, "default": "../../config/config.json",
                      "help": "path to the config file"},
                     {"name": "--obj_asset_root", "type": str, "default": "../../data",
                      "help": "root path to object urdfs"},
                     {"name": "--panda_asset_root", "type": str, "default": "../../assets",
                      "help": "root path to panda urdf"},
                     {"name": "--panda_asset_file", "type": str, "default": "panda.urdf",
                      "help": "child path to panda urdf"},
                     {"name": "--vpn_panda_mesh", "type": str, "default": "../../assets/panda_gripper_col4.ply",
                      "help": "used for gprts in vpn"},
                     {"name": "--idx", "type": int, "default": 1, "help": "save file idx"},
                     ]

# Add custom arguments
# parse arguments
args = gymutil.parse_arguments(description="Body Physics Properties Example",
                               custom_parameters=custom_parameters, )

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '%s'%args.gpu
device = 'cuda'
import json
with open(args.config, 'r') as config_file:
    cfg = json.load(config_file)
import sys
# sys.path.append('/home/sujc/catkin_ws/src/vcpd')
# sys.path.append('/home/sujc/catkin_ws/src/vcpd/src')
from isaacgym import gymapi
from isaacgym import gymtorch

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from isaac.utils import get_tray, add_noise, PandaGripper
import trimesh
from vpn_sdf import TSDF
from isaac.utils import vpn_predict
from vpn.model import VPN, RefineNetV0
from vpn.utils import DiscretizedGripper
vpn = VPN()
if args.rn_used == 0:
    print('use vpn')
else:
    print('use vpn and rn')
rn = RefineNetV0(num_sample=cfg['refine']['num_sample'])
vpn.load_network_state_dict(device=device, pth_file=args.vpn_path)
rn.load_network_state_dict(device=device, pth_file=args.rn_path)
vpn.to(device)
rn.to(device)
dg = DiscretizedGripper(cfg['refine'])


def load_panda(sim, args):
    if args.panda_asset_root is None:
        asset_root = "/home/sujc/code/isaacgym/assets"
        panda_asset_file = "urdf/franka_description/robots/panda.urdf"
    else:
        asset_root = args.panda_asset_root
        panda_asset_file = args.panda_asset_file
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    # asset_options.override_com = True
    # asset_options.override_inertia = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    panda_asset = gym.load_asset(sim, asset_root, panda_asset_file, asset_options)
    obj_prop = gymapi.RigidShapeProperties()
    obj_prop.friction = 3.0
    obj_prop.restitution = 0.9
    gym.set_asset_rigid_shape_properties(panda_asset, [obj_prop])
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

    return default_dof_pos, default_dof_state, panda_asset, panda_dof_props


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
        asset_options.vhacd_enabled = True
    # idxs = np.random.choice(len(asset_path), num_objects, replace=False)
    # for idx in idxs:
    #     asset = asset_path[idx]
    for asset in asset_path:
        if '168' in asset:
            continue
        print("Loading asset '%s' from '%s'" % (asset, asset_root))
        current_asset = gym.load_asset(sim, asset_root, asset, asset_options)
        if current_asset is None:
            print("*** Failed to load asset '%s'" % (asset))
            quit()
        obj_prop = gymapi.RigidShapeProperties()
        obj_prop.friction = 5.0
        # obj_prop.restitution = 0.9
        obj_prop.rolling_friction = 3.0
        gym.set_asset_rigid_shape_properties(current_asset, [obj_prop])
        loaded_assets.append(current_asset)
        # count += 1
        # if count == args.num_objects:
        #     break
    return loaded_assets


def create_obj_assets(args):
    type = args.obj_type
    if type == 'primitive' or type == 0:
        if args.obj_asset_root is None:
            asset_root = '../../data/primitive/primitive_urdf'
        else:
            asset_root = args.obj_asset_root + '/primitive_urdf'
        paths = os.listdir(asset_root)
        asset_paths = []
        for path in paths:
            asset_paths.append(path + '/' + path + '.urdf')
        obj_assets = load_obj(gym, sim, asset_root, asset_paths)

    elif type == 'random' or type == 1:
        if args.obj_asset_root is None:
            asset_root = '../../data/random_urdfs'
        else:
            asset_root = args.obj_asset_root + '/random_urdfs'
        paths = os.listdir(asset_root)
        asset_paths = []
        for path in paths:
            asset_paths.append(path + '/' + path + '.urdf')
        obj_assets = load_obj(gym, sim, asset_root, asset_paths)

    elif type == 'kit' or type == 2:
        if args.obj_asset_root is None:
            asset_root = '../../data/kit_urdf'
        else:
            asset_root = args.obj_asset_root + '/kit_urdf'
        paths = os.listdir(asset_root)
        asset_paths = []
        for path in paths:
            asset_paths.append(path + '/' + path + '.urdf')
        obj_assets = load_obj(gym, sim, asset_root, asset_paths)

    # simple objects
    elif type == 'simple' or type == 3:
        box_size = 0.045
        asset_options = gymapi.AssetOptions()
        obj_assets = []
        for i in range(args.num_objects):
            obj_assets.append(gym.create_box(sim, box_size, box_size, box_size, asset_options))
            obj_prop = gymapi.RigidShapeProperties()
            obj_prop.friction = 3.0
            # obj_prop.restitution = 0.9
            obj_prop.rolling_friction = 3.0
            gym.set_asset_rigid_shape_properties(obj_assets[-1], [obj_prop])

    # egad objects
    elif type == 'egad' or type == 4:
        asset_root = 'data/train/train_urdf'
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


def get_pose(env, cam_handle):
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

# initialize gym
gym = gymapi.acquire_gym()

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

# create viewer
if not args.headless:
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

spacing = 0.8
# spacing = cfg['sdf']['x_max']
env_lower = gymapi.Vec3(-spacing, -spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, cfg['sdf']['z_max'])

# load bin asset-------------------------------------------------------------
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
num_objects = args.num_objects
asset_root = "/home/sujc/code/isaacgym/assets"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX


size_list, pose_list = get_tray()
bins = []
for i in range(len(size_list)):
    bins.append(gym.create_box(sim, size_list[i][0], size_list[i][1], size_list[i][2], asset_options))
# create static box asset
asset_options.density = 1000000
box_z = 1.5
asset_box = gym.create_box(sim, 0.6, 0.01, box_z, asset_options)
asset_cam_body = gym.create_box(sim, 0.0001, 0.0001, 0.0001, asset_options)

# load panda gripper
default_dof_pos, default_dof_state, \
panda_asset, panda_dof_props = load_panda(sim, args)

# load object assets -----------------------------------------------------------
obj_assets = create_obj_assets(args)

envs = []

tray_handles = []

obj_handles = []
obj_root_idxs = []

panda_handles = []
panda_root_idxs = []

box_handles = []
box_root_idxs = []

cams = []
cam_root_idxs = []

depth_tensors = []
color_tensors = []

cam_body_handles = []
cam_body_root_idxs = []

# set cam props
get_intrinsic = False
cam_z = 0.55
cam_props = gymapi.CameraProperties()
cam_props.width = 640
cam_props.height = 480
cam_props.enable_tensors = True
cam_props.use_collision_geometry = False
print('Creating %d environments' % num_envs)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

    for ii in range(len(size_list)):
        tray_handles.append(gym.create_actor(env, bins[ii], pose_list[ii], "bin" + str(ii), i, 0))
        gym.set_rigid_body_color(env, tray_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    obj_list = np.random.choice(obj_assets, args.num_objects, replace=len(obj_assets)<args.num_objects)

    # create 5 boxes
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.3, box_z / 2)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    box_handle = gym.create_actor(env, asset_box, pose, str(i) + "box1", i, 0)
    box_handles.append(box_handle)
    box_root_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
    box_root_idxs.append(box_root_idx)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, -0.3, box_z / 2)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    box_handle = gym.create_actor(env, asset_box, pose, str(i) + "box2", i, 0)
    box_handles.append(box_handle)
    box_root_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
    box_root_idxs.append(box_root_idx)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.3, 0.0, box_z / 2)
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0.5 * math.pi)
    box_handle = gym.create_actor(env, asset_box, pose, str(i) + "box3", i, 0)
    box_handles.append(box_handle)
    box_root_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
    box_root_idxs.append(box_root_idx)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(-0.3, 0.0, box_z / 2)
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0.5 * math.pi)
    box_handle = gym.create_actor(env, asset_box, pose, str(i) + "box4", i, 0)
    box_handles.append(box_handle)
    box_root_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
    box_root_idxs.append(box_root_idx)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, box_z)
    pose.r = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    box_handle = gym.create_actor(env, asset_box, pose, str(i) + "box5", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    box_handles.append(box_handle)
    box_root_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
    box_root_idxs.append(box_root_idx)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1e-3, 0.0, cam_z)
    pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(90))
    cam_body_handle = gym.create_actor(env, asset_cam_body, pose, str(i) + "cam_body", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, cam_body_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    cam_body_handles.append(cam_body_handle)
    cam_body_root_idx = gym.get_actor_index(env, cam_body_handle, gymapi.DOMAIN_SIM)
    cam_body_root_idxs.append(cam_body_root_idx)

    # add panda
    panda_pose = gymapi.Transform()
    panda_pose.p = gymapi.Vec3(1, 1, 1)
    panda_pose.r = gymapi.Quat.from_euler_zyx(0, math.pi, 0)
    a = R.from_euler('zyx', [np.pi, 0, 0]).as_matrix()
    gymapi.Quat()
    panda_handle = gym.create_actor(env, panda_asset, panda_pose, "panda_%s" % i, i, 0)
    panda_handles.append(panda_handle)

    gym.set_actor_dof_properties(env, panda_handle, panda_dof_props)
    gym.set_actor_dof_states(env, panda_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, panda_handle, default_dof_pos)

    hand_handle = gym.find_actor_rigid_body_handle(env, panda_handle, "panda_hand")
    # get global root index of obj in rigid body state tensor
    panda_root_idx = gym.get_actor_index(env, hand_handle, gymapi.DOMAIN_SIM)
    panda_root_idxs.append(panda_root_idx)

    # create objects
    for j in range(args.num_objects):
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1),
                                 np.random.uniform(0.05, 0.2))
        obj_pose.r = gymapi.Quat.from_euler_zyx(np.random.rand() * math.pi,
                                                np.random.rand() * math.pi,
                                                np.random.rand() * math.pi)
        obj_handle = gym.create_actor(env, obj_list[j], obj_pose, str(i) + str(j), i, 0)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        body_props = gym.get_actor_rigid_body_properties(env, obj_handle)
        body_props[0].mass = 0.05
        body_props[0].flags = 0
        gym.set_actor_rigid_body_properties(env, obj_handle, body_props)
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

# position viewer camera
if not args.headless:
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

gym.prepare_sim(sim)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 2, 1)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

# get root body tensor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)

vol_bnd = np.array([cfg['sdf']['x_min'], cfg['sdf']['y_min'], cfg['sdf']['z_min'] - 0.01,
                    cfg['sdf']['x_max'], cfg['sdf']['y_max'], cfg['sdf']['z_max'] - 0.01]).reshape(2, 3)
voxel_length = cfg['sdf']['voxel_lengthresolution']

tsdf = TSDF(vol_bnd.T, voxel_length, rgb=False, device=device)
pg = PandaGripper('../../assets')
grasp_poses = []
grasps_widths = []
grasp_rots = []
grasp_quats = []
depths = []
cam_poses = []
init_d = 0.5
delta_d = 0.004
s = stable = 500
f = cfg['test']['frame']
d = down = np.round(init_d / delta_d)
u = up = 400
s_ = shake = 30
a = away = 450
c = close = 120
t = 0

time = {'stable': s,
        'tsdf': s + f,
        'down': s + f + d,
        'close': s + f + d + c,
        'up': s + f + d + c + u,
        'away': s + f + d + c + u + a,
        'shake_1': s + f + d + c + u + a + 1 * s_,
        'shake_2': s + f + d + c + u + a + 2 * s_,
        'shake_3': s + f + d + c + u + a + 3 * s_,
        'shake_4': s + f + d + c + u + a + 4 * s_,
        'open': s + f + d + 2 * c + u + a + 6 * s_,
        }

isaac_rot = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
isaac_rot_inv = np.linalg.inv(isaac_rot)
p1 = gymapi.Transform()
p2 = gymapi.Transform()
count = 0
succ = 0

# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.5)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
# radius :0.01
sphere_geom_1 = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=(1, 0, 0))
sphere_geom_2 = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=(0, 1, 0))
# radius :0.005
sphere_geom_1s = gymutil.WireframeSphereGeometry(0.005, 12, 12, sphere_pose, color=(1, 1, 0))
sphere_geom_2s = gymutil.WireframeSphereGeometry(0.005, 12, 12, sphere_pose, color=(0, 1, 1))

def step(sim):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

def visual_cpts(cp1s ,cp2s,sphere_geom_1s,sphere_geom_2s,env,clear=False):
    if clear:
        gym.clear_lines(viewer)
    for j in range(cp1s.shape[0]):
        cp1 = cp1s[j]
        cp2 = cp2s[j]
        color = gymapi.Vec3(1, 0, 0)
        p1.p = gymapi.Vec3(cp1[0], cp1[1], cp1[2])
        p2.p = gymapi.Vec3(cp2[0], cp2[1], cp2[2])
        gymutil.draw_line(p1.p, p2.p, color, gym, viewer, env)
        gymutil.draw_lines(sphere_geom_1s, gym, viewer, env, p1)
        gymutil.draw_lines(sphere_geom_2s, gym, viewer, env, p2)
continue_num = 0
grasp_attempts = min(args.grasp_per_env, num_objects)
while True:
    if not args.headless:
        if gym.query_viewer_has_closed(viewer):
            break

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)

    if t == time['stable'] - 1:
        rigid_obj = torch.clone(root_tensor)
        rigid_obj[box_root_idxs, :2] = -10
        rigid_obj[box_root_idxs, 2] = 0.01 / 2
        rigid_obj[box_root_idxs, 3:7] = 0
        rigid_obj[box_root_idxs, 3] = rigid_obj[box_root_idxs, 6] = 0.70107
        actor_indices = torch.tensor([box_root_idxs], dtype=torch.int32, device=device)
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                gymtorch.unwrap_tensor(actor_indices),
                                                len(box_root_idxs))

    if t == time['stable']:
        init_rot, w2c = get_pose(envs[-1], cams[-1])

    # update the viewer
    gym.step_graphics(sim)

    # render sensors and refresh camera tensors
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    if time['stable'] < t <= time['tsdf']:
        depth_s = []
        mass = []
        for i in range(num_envs):
            for handle in obj_handles:
                body_props = gym.get_actor_rigid_body_properties(envs[0], handle)[0]
                mass.append(body_props.mass)
            depth = -torch.clone(depth_tensors[i]).cpu().numpy()
            depth[depth == -np.inf] = 0
            depth = add_noise(depth, intrinsic)
            depth_s.append(depth)
        depths.append(depth_s)
        cam_poses.append(w2c @ isaac_rot_inv)

        cam_pos, cam_rot, w2c = sample_a_pose_from_a_sphere(0.65, np.array([0, 0, 0]))
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
                tsdf.tsdf_integrate(depths[j][i], intrinsic, cam_poses[j])
            # tsdf.write_mesh('out_%s.ply' % i, *tsdf.compute_mesh(step_size=2))

            use_rn = False
            if args.rn_used == 0:
                use_rn = True
            panda_gripper_mesh = trimesh.load_mesh(args.vpn_panda_mesh)
            gpr_pts = torch.from_numpy(panda_gripper_mesh.vertices.astype(np.float32)).to(device)
            pos, rot, quat = vpn_predict(tsdf, vpn, dg, rn, gpr_pts, device, use_rn=use_rn)
            if pos is None:
                t = time['up'] + 20
                continue_num += 1
                print('env %s ,continue %s'%(i, continue_num))
                if continue_num == 2:
                    pos = np.array([0,0,1])
                    quat = np.array([1,0,0,0])
                    rot = R.from_quat(quat).as_matrix()
                else:
                    continue
            grasp_poses.append(pos)
            grasp_rots.append(rot)
            grasp_quats.append(quat)
            # grasps_widths.append(width)
            tsdf.reset()

        grasp_poses = np.array(grasp_poses)
        grasp_rots = np.array(grasp_rots)
        # grasps_widths = np.array(grasps_widths)
        grasp_quats = np.array(grasp_quats)
        grasp_quats = torch.from_numpy(grasp_quats).to(device)


    if time['tsdf'] < t <= time['down']:
        rigid_obj = torch.clone(root_tensor)
        grasp_poses_ = grasp_poses - grasp_rots[..., 2] * (init_d - (t - time['tsdf']) * delta_d - 0.005)
        grasp_poses_ = torch.from_numpy(grasp_poses_).to(device)

        rigid_obj[panda_root_idxs, :3] = grasp_poses_.to(torch.float32)
        rigid_obj[panda_root_idxs, 3:7] = grasp_quats.to(torch.float32)
        actor_indices = torch.tensor([panda_root_idxs], dtype=torch.int32, device=device)
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                gymtorch.unwrap_tensor(actor_indices),
                                                num_envs)

    if t == time['down']:
        pos_action[:] = get_gripper_dof(num_envs, action='close', device=device)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
        obj_z_1 = torch.clone(root_tensor[obj_root_idxs,2])
    gym.end_access_image_tensors(sim)

    if time['close'] < t <= time['up']:
        rigid_obj = torch.clone(root_tensor)
        rigid_obj[panda_root_idxs, 2] += 0.0005
        actor_indices = torch.tensor([panda_root_idxs], dtype=torch.int32, device=device)
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                gymtorch.unwrap_tensor(actor_indices),
                                                num_envs)

    if t == time['up']:
        count += num_envs
        success = (dof_pos.squeeze(-1).clone()[..., -1] > 0.003).cpu().numpy()
        succ += success.sum()
        print('succ:%s, grasps:%s, rate:%.2f%%' % (succ, count, succ / count * 100))

        obj_z_2 = torch.clone(root_tensor[obj_root_idxs, 2])
        obj_diff_z = (obj_z_2-obj_z_1).reshape(num_envs, -1).cpu().numpy()
        obj_idx = np.argmax(obj_diff_z, axis=-1)
        obj_root_idxs_ = np.array(obj_root_idxs).reshape(num_envs, -1)
        idxs = np.stack([np.arange(num_envs), obj_idx])
        obj_root_idx = obj_root_idxs_[idxs[0], idxs[1]].tolist()

        rigid_obj = torch.clone(root_tensor)
        rigid_obj[obj_root_idx, 1] = 0.6
        actor_indices = torch.tensor([obj_root_idx], dtype=torch.int32, device=device)
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                gymtorch.unwrap_tensor(actor_indices),
                                                num_envs)

        rigid_obj[panda_root_idxs, 2] = 0.7
        actor_indices = torch.tensor(panda_root_idxs, dtype=torch.int32, device=device)
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                gymtorch.unwrap_tensor(actor_indices),
                                                num_envs)

    if t == time['up']+1:
        pos_action[:] = get_gripper_dof(num_envs, action='open', device=device)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    if t == time['up']+20:
        t = 0
        if not args.headless:
            gym.clear_lines(viewer)
        grasp_poses = []
        grasps_widths = []
        grasp_rots = []
        grasp_quats = []
        depths = []
        cam_poses = []
        tsdf.reset()

    t += 1

    if not args.headless:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        if gym.query_viewer_has_closed(viewer):
            break

    if count == num_envs * grasp_attempts:
        break
print('Done')
txt = np.array([succ, count]).astype(np.int)
if args.rn_used == 0:
    path = '../log/vpn'
else:
    path = '../log/rn'
if not os.path.exists(path):
    os.makedirs(path)
name = ['primitive', 'random', 'kit']
np.savetxt(path+'/%s_%s_%s.txt'%(name[args.obj_type], num_objects, args.idx), txt, fmt='%i', delimiter=",")
if not args.headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
bb=tt.time()
print('time cost:{:0.2f} min'.format((bb-aa)/60))