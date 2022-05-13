from sim.utils import step_simulation
from scipy.spatial.transform import Rotation
import pybullet as p
import numpy as np
import os


class RigidObject(object):
    def __init__(self, obj_name, **kwargs):
        """
        Construct a rigid object for pybullet.
        :param obj_name: a string of object name.
        :param vis_params: parameters of p.createVisualShape.
        :param col_params: parameters of p.createCollisionShape.
        :param body_params: parameters of p.createMultiBody.
        """
        self.obj_name = obj_name
        keys = kwargs.keys()
        if 'vis_params' in keys and 'col_params' in keys and 'body_params' in keys:
            vis_params, col_params, body_params = kwargs['vis_params'], kwargs['col_params'], kwargs['body_params']
            self.obj_id = p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(**col_params),
                                            baseVisualShapeIndex=p.createVisualShape(**vis_params),
                                            **body_params)
        elif 'fileName' in keys:
            self.obj_id = p.loadURDF(kwargs['fileName'],
                                     basePosition=kwargs['basePosition'], baseOrientation=kwargs['baseOrientation'])
            p.changeDynamics(self.obj_id, linkIndex=-1, mass=kwargs['mass'])
        else:
            raise ValueError('Invalid arguments for RigidObject initialization.')

    def change_dynamics(self, *args, **kwargs):
        p.changeDynamics(self.obj_id, -1, *args, **kwargs)

    def change_color(self, rgb=None, a=1.0):
        if rgb is None:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=np.random.uniform(size=3).tolist()+[a])
        else:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=np.asarray(rgb).tolist() + [a])

    def get_pose(self):
        position, quaternion = p.getBasePositionAndOrientation(self.obj_id)
        return np.asarray(position), np.asarray(quaternion)

    def set_pose(self, position, quaternion):
        p.resetBasePositionAndOrientation(self.obj_id,
                                          position,
                                          quaternion)

    def wait_for_stable_condition(self, threshold=0.001, num_step=30):
        stable = False
        while not stable:
            if self.is_stable(threshold, num_step):
                stable = True

    def is_stable(self, threshold=0.001, num_steps=30):
        pose1 = np.concatenate(self.get_pose(), axis=0)
        step_simulation(num_steps)
        pose2 = np.concatenate(self.get_pose(), axis=0)
        fluctuation = np.sqrt(np.sum((np.asanyarray(pose1) - np.asanyarray(pose2)) ** 2))
        # print('fluctuation: {}'.format(fluctuation))
        return fluctuation < threshold

    @property
    def transform(self):
        pos, quat = self.get_pose()
        pose = np.eye(4, dtype=np.float32)
        pose[0:3, 0:3] = Rotation.from_quat(quat).as_matrix()
        pose[0:3, 3] = pos
        return pose


class PandaGripper(object):
    def __init__(self, asset_path):
        self.components = ['hand', 'left_finger', 'right_finger']
        self.vertex_sets = dict()
        for component in self.components:
            vis_path = os.path.join(asset_path, component+'.obj')
            col_path = os.path.join(asset_path, component+'_col.obj')
            vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [1] * 3}
            col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [1] * 3}
            body_params = {'baseMass': 0, 'basePosition': [0, 0, 0], 'baseOrientation': [0, 0, 0, 1]}
            self.__setattr__(component, RigidObject(component,
                                                    vis_params=vis_params,
                                                    col_params=col_params,
                                                    body_params=body_params))
        self._max_width = 0.08
        self._curr_width = 0.08

    def set_pose(self, position, quaternion):
        for com in self.components:
            self.__getattribute__(com).set_pose(position, quaternion)
        curr_width = self._curr_width
        self._curr_width = self._max_width
        self.set_gripper_width(curr_width)

    def get_pose(self):
        return self.__getattribute__(self.components[0]).get_pose()

    def set_gripper_width(self, width):
        # if width > self._max_width:
        #     print('warning! the maximal width is 0.08 for panda gripper, set to 0.08 instead')
        #     width = self._max_width
        width = min(0.08, max(0.0, width))
        left_delta = np.eye(4)
        left_delta[1, 3] = (width - self._curr_width) / 2
        left_pose = self.left_finger.transform @ left_delta
        self.left_finger.set_pose(left_pose[0:3, 3], Rotation.from_matrix(left_pose[0:3, 0:3]).as_quat())
        right_delta = np.eye(4)
        right_delta[1, 3] = -(width - self._curr_width) / 2
        right_pose = self.right_finger.transform @ right_delta
        self.right_finger.set_pose(right_pose[0:3, 3], Rotation.from_matrix(right_pose[0:3, 0:3]).as_quat())
        self._curr_width = width

    def get_gripper_width(self):
        return self._curr_width

    def reset_gripper(self):
        self.set_pose([0, 0, 0], [0, 0, 0, 1])
        self.set_gripper_width(self._max_width)

    def is_collided(self, exemption, threshold=0.0, show_col=False):
        gripper_ids = [self.__getattribute__(com).obj_id for com in self.components]
        for obj_id in range(p.getNumBodies()):
            if obj_id not in gripper_ids and obj_id not in exemption:
                for gripper_id in gripper_ids:
                    contacts = p.getClosestPoints(gripper_id, obj_id, threshold)
                    if len(contacts) != 0:
                        p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1]) if show_col else None
                        return True
        return False

    def change_color(self, rgb=None, a=1.0):
        if rgb is None:
            rgba = np.random.uniform(size=3).tolist()+[a]
        else:
            rgba = np.asarray(rgb).tolist()+[a]
        [p.changeVisualShape(self.__getattribute__(com).obj_id, -1, rgbaColor=rgba) for com in self.components]

    def remove_gripper(self):
        [p.removeBody(self.__getattribute__(com).obj_id) for com in self.components]
