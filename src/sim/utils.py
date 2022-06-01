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
                         zero, sin, cos], axis=-1).reshape(shape+[3, 3])
    elif axis == 'y':
        rots = np.stack([cos, zero, sin,
                         zero, one, zero,
                         -sin, zero, cos], axis=-1).reshape(shape+[3, 3])
    elif axis == 'z':
        rots = np.stack([cos, -sin, zero,
                         sin, cos, zero,
                         zero, zero, one], axis=-1).reshape(shape+[3, 3])
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
    s = add_sphere(pos, radius=0.00001)
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


def get_contact_points(cp1, cp2, obj):
    grasp_direction = cp1 - cp2
    grasp_center = (cp1 + cp2) / 2
    grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
    length = 0.001
    half_w = 0.07
    intersections = p.rayTestBatch([cp1+length*grasp_direction, cp2-length*grasp_direction],
                                   [grasp_center, grasp_center])
    while length < half_w and np.logical_or(intersections[0][0] == -1, intersections[1][0] == -1):
        length += 0.001
        intersections = p.rayTestBatch([cp1+length*grasp_direction, cp2-length*grasp_direction],
                                       [grasp_center, grasp_center])
        # l1 = p.addUserDebugLine(cp1+length*grasp_direction, grasp_center, lineColorRGB=[1, 0, 0])
        # l2 = p.addUserDebugLine(cp2-length*grasp_direction, grasp_center, lineColorRGB=[0, 1, 0])
        # p.removeUserDebugItem(l1), p.removeUserDebugItem(l2)
    if abs(length - half_w) < 1e-7:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        print('no contact point found, return None.')
        return None, None, None, None
    contact1, normal1 = np.asarray(intersections[0][3]), np.asarray(intersections[0][4])
    contact2, normal2 = np.asarray(intersections[1][3]), np.asarray(intersections[1][4])
    if intersections[0][0] != obj.obj_id or intersections[1][0] != obj.obj_id:
        print('Warning! The retrieved contact points do not belong to the target object.')
    return contact1, normal1, contact2, normal2


def get_contact_points_from_center(grasp_center, grasp_direction, obj):
    length = 0.001
    half_w = 0.07
    intersections = p.rayTestBatch([grasp_center+length*grasp_direction, grasp_center-length*grasp_direction],
                                   [grasp_center, grasp_center])
    while length < half_w and np.logical_or(intersections[0][0] == -1, intersections[1][0] == -1):
        length += 0.001
        intersections = p.rayTestBatch([grasp_center+length*grasp_direction, grasp_center-length*grasp_direction],
                                       [grasp_center, grasp_center])
    if abs(length - half_w) < 1e-7:
        print('no contact point found, return None.')
        return None, None, None, None
    contact1, normal1 = np.asarray(intersections[0][3]), np.asarray(intersections[0][4])
    contact2, normal2 = np.asarray(intersections[1][3]), np.asarray(intersections[1][4])
    if intersections[0][0] != obj.obj_id or intersections[1][0] != obj.obj_id:
        print('Warning! The retrieved contact points do not belong to the target object.')
    return contact1, normal1, contact2, normal2


def visualize_contacts(cp1, cp2, num_vis=50):
    """
    visualize contact points in pybullet.
    :param cp1: A Nx3-d numpy array representing one side of contact points
    :param cp2: A Nx3-d numpy array representing another side of contact points
    :param num_vis: the number of contact pairs that visualize in the pybullet.
    :return: None
    """
    assert isinstance(cp1, np.ndarray)
    ids = np.random.choice(np.arange(cp1.shape[0]), min(num_vis, cp1.shape[0]))
    selected_cp1, selected_cp2 = cp1[ids], cp2[ids]
    radius = 0.003
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    cp1s = [p.createMultiBody(0,
                              p.createCollisionShape(p.GEOM_SPHERE, radius),
                              p.createVisualShape(p.GEOM_SPHERE, radius, rgbaColor=[220/255, 20/255, 60/255, 1]),
                              basePosition=cp) for cp in selected_cp1]
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    cp2s = [p.createMultiBody(0,
                              p.createCollisionShape(p.GEOM_SPHERE, radius),
                              p.createVisualShape(p.GEOM_SPHERE, radius, rgbaColor=[60/255, 179/255, 113/255, 1]),
                              basePosition=cp) for cp in selected_cp2]
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    lines = [p.addUserDebugLine(selected_cp1[pid], selected_cp2[pid],
                                lineColorRGB=np.random.uniform(size=3),
                                lineWidth=0.1) for pid in range(selected_cp2.shape[0])]
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    [p.removeBody(cp) for cp in cp1s]
    [p.removeBody(cp) for cp in cp2s]
    [p.removeUserDebugItem(line) for line in lines]
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def visualize_contact(cp1, n1, cp2, n2):
    l0 = p.addUserDebugLine(cp1, cp2)
    s1 = add_sphere(cp1, radius=0.004, rgb=[220/255, 20/255, 60/255])
    l1 = p.addUserDebugLine(cp1 - 0.01 * n1, cp1 + 0.01 * n1, lineColorRGB=[220/255, 20/255, 60/255], lineWidth=0.2)
    s2 = add_sphere(cp2, radius=0.004, rgb=[60/255, 179/255, 113/255])
    l2 = p.addUserDebugLine(cp2 - 0.01 * n2, cp2 + 0.01 * n2, lineColorRGB=[60/255, 179/255, 113/255], lineWidth=0.2)
    input('press Enter to remove the contact')
    p.removeBody(s1), p.removeBody(s2)
    p.removeUserDebugItem(l0), p.removeUserDebugItem(l1), p.removeUserDebugItem(l2)
