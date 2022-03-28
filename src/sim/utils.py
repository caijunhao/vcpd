import pybullet as p
import numpy as np


def get_multi_body_template():
    vis_params = {'shapeType': p.GEOM_MESH,
                  'fileName': '',
                  'meshScale': [1, 1, 1]}
    col_params = {'shapeType': p.GEOM_MESH,
                  'fileName': '',
                  'meshScale': [1, 1, 1]}
    body_params = {'baseMass': 0,
                   'basePosition': [0, 0, 0],
                   'baseOrientation': [0, 0, 0, 1]}
    return vis_params, col_params, body_params


def sample_a_pose(x_min, x_max,
                  y_min, y_max,
                  z_min, z_max,
                  alpha_min, alpha_max,
                  beta_min, beta_max,
                  gamma_min, gamma_max):
    random_x = np.random.uniform(x_min, x_max)
    random_y = np.random.uniform(y_min, y_max)
    random_z = np.random.uniform(z_min, z_max)
    random_alpha = np.random.uniform(alpha_min, alpha_max)
    random_beta = np.random.uniform(beta_min, beta_max)
    random_gamma = np.random.uniform(gamma_min, gamma_max)
    position = np.array([random_x, random_y, random_z])
    quaternion = np.array(p.getQuaternionFromEuler([random_alpha, random_beta, random_gamma]))
    return position, quaternion


def basic_rot_mat(angles, axis):
    cos = np.cos(angles)
    sin = np.sin(angles)
    zero = np.zeros_like(cos)
    one = np.ones_like(cos)
    shape = list(cos.shape)
    if axis == 'x':
        rots = np.stack([one, zero, zero,
                         zero, cos, -sin,
                         zero, sin, cos], axis=1).reshape(shape+[3, 3])
    elif axis == 'y':
        rots = np.stack([cos, zero, sin,
                         zero, one, zero,
                         -sin, zero, cos], axis=1).reshape(shape+[3, 3])
    elif axis == 'z':
        rots = np.stack([cos, -sin, zero,
                         sin, cos, zero,
                         zero, zero, one], axis=1).reshape(shape+[3, 3])
    else:
        raise ValueError('invalid axis')
    return rots


def step_simulation(num_steps):
    for i in range(num_steps):
        p.stepSimulation()


def add_sphere(pos, radius=0.002, rgb=None):
    obj_id = p.createMultiBody(0,
                               p.createCollisionShape(p.GEOM_SPHERE, radius),
                               p.createVisualShape(p.GEOM_SPHERE, radius),
                               basePosition=pos)
    if rgb:
        p.changeVisualShape(obj_id, -1, rgbaColor=np.asarray(rgb).tolist()+[1])
    return obj_id


def get_closest_obj(pos, obj_list):
    s = add_sphere(pos)
    closest = 1e7
    closest_obj = obj_list[0]
    for obj in obj_list:
        contacts = p.getClosestPoints(s, obj.obj_id, distance=0.04)
        if len(contacts) != 0:
            min_dist = min(contacts, key=lambda x: abs(x[8]))[8]
            if min_dist < closest:
                closest_obj = obj
                closest = min_dist
    p.removeBody(s)
    return closest_obj


def get_closest_contact(pos, obj):
    s = add_sphere(pos, radius=0.0005)
    th = 0.005
    contact, normal = None, None
    while contact is None:
        contacts = p.getClosestPoints(s, obj.obj_id, distance=th)
        if len(contacts) != 0:
            contact, normal = min(contacts, key=lambda x: abs(x[8]))[6:8]
        else:
            th += 0.005
    p.removeBody(s)
    return np.array(contact), np.array(normal)


def check_valid_pts(pts, bounds):
    """
    check if the given points are inside the boundaries or not.
    :param pts: an Nx3-d numpy array representing the 3-d points.
    :param bounds: a 2x3-d numpy array representing the boundaries of the volume.
    :return: an N-d numpy array representing if the points are inside the given boundaries or not.
    """
    flag_x = np.logical_and(pts[:, 0] >= bounds[0, 0], pts[:, 0] <= bounds[1, 0])
    flag_y = np.logical_and(pts[:, 1] >= bounds[0, 1], pts[:, 1] <= bounds[1, 1])
    flag_z = np.logical_and(pts[:, 2] >= bounds[0, 2], pts[:, 2] <= bounds[1, 2])
    flag = flag_x * flag_y * flag_z
    return flag
