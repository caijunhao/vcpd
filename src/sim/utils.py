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
