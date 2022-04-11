import pybullet as p
import numpy as np


class Tray(object):
    def __init__(self, height, width, depth, theta=45, thickness=0.01):
        """
        Create a static (mass=0) tray.
        :param height: Height of the workspace in the groove.
        :param width: Width of the workspace in the groove.
        :param depth: Depth of the groove.
        :param theta: Slope of the bevel of the groove.
        :param thickness: Thickness for each part of the groove.
        """
        d2r = np.deg2rad
        h, w, d, t = height / 2, width / 2, depth / 2, thickness / 2
        assert 0 < theta <= 90
        side_height = depth / np.sin(d2r(theta))
        s_h = side_height / 2
        e = 0 if theta == 90 else depth / np.tan(d2r(theta))  # half extend length for each side of groove
        # compute the normal and tangent components of half thickness
        n_com, t_com = t * np.cos(d2r(theta)), t * np.sin(d2r(theta))
        self.piece0 = p.createMultiBody(0,
                                        p.createCollisionShape(p.GEOM_BOX, halfExtents=[h, w, t]),
                                        basePosition=[0, 0, t])
        self.piece1 = p.createMultiBody(0,
                                        p.createCollisionShape(p.GEOM_BOX, halfExtents=[h + e, s_h, t]),
                                        basePosition=[0, -w - e / 2 - t_com, d + thickness - n_com],
                                        baseOrientation=p.getQuaternionFromEuler([d2r(-theta), d2r(0), d2r(0)]))
        self.piece2 = p.createMultiBody(0,
                                        p.createCollisionShape(p.GEOM_BOX, halfExtents=[h + e, s_h, t]),
                                        basePosition=[0, w + e / 2 + t_com, d + thickness - n_com],
                                        baseOrientation=p.getQuaternionFromEuler([d2r(theta), d2r(0), d2r(0)]))
        self.piece3 = p.createMultiBody(0,
                                        p.createCollisionShape(p.GEOM_BOX, halfExtents=[s_h, w + e, t]),
                                        basePosition=[-h - e / 2 - t_com, 0, d + thickness - n_com],
                                        baseOrientation=p.getQuaternionFromEuler([d2r(0), d2r(theta), d2r(0)]))
        self.piece4 = p.createMultiBody(0,
                                        p.createCollisionShape(p.GEOM_BOX, halfExtents=[s_h, w + e, t]),
                                        basePosition=[h + e / 2 + t_com, 0, d + thickness - n_com],
                                        baseOrientation=p.getQuaternionFromEuler([d2r(0), -d2r(theta), d2r(0)]))
        self.base = np.array([0, 0, t])
        self.change_color()

    def change_color(self, a=0.7):
        rgb = [np.random.uniform(127 / 255, 248 / 255)] * 3
        for name in self.__dict__.keys():
            if 'piece' in name:
                p.changeVisualShape(self.__getattribute__(name), -1, rgbaColor=rgb + [a])

    def set_pose(self, position):
        curr_base_pos, _ = p.getBasePositionAndOrientation(self.__getattribute__('piece0'))
        curr_base_pos = np.array(curr_base_pos)
        for name in self.__dict__.keys():
            if 'piece' in name:
                pos, quat = p.getBasePositionAndOrientation(self.__getattribute__(name))
                pos = np.array(pos) + np.array(position) - curr_base_pos + self.__getattribute__('base')
                quat = np.array(quat)
                p.resetBasePositionAndOrientation(self.__getattribute__(name), pos, quat)

    def get_tray_ids(self):
        return [self.__getattribute__(name) for name in self.__dict__.keys() if 'piece' in name]
