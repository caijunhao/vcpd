from scipy.spatial.transform import Rotation
from sim.utils import basic_rot_mat
from sim.objects import PandaGripper
import numpy as np


def rotate_gripper(end_point_pos, rot, width, griper_depth, exemptions):
    """
    used for drawing figure
    """
    pg = PandaGripper('assets')
    basic_rots = basic_rot_mat(np.arange(16) / 16 * np.pi * 2, axis='y')
    rots = np.matmul(rot.reshape((1, 3, 3)), basic_rots)
    pg_config = []
    for i in range(16):
        quat_i = Rotation.from_matrix(rots[i]).as_quat()
        pos_g = end_point_pos - rots[i][:, 2] * griper_depth
        pg.set_pose(pos_g, quat_i)
        pg.set_gripper_width(width)
        col = pg.is_collided(exemptions)
        if col:
            pg_config.append((pos_g, quat_i, width, [220 / 255, 20 / 255, 60 / 255], 0.3))
        else:
            pg_config.append((pos_g, quat_i, width, [1, 1, 1], 0.5))
    pg.remove_gripper()
    pgs = list()
    for config in pg_config:
        pos, quat, w, rgb, a = config
        pgs.append(PandaGripper('assets'))
        pgs[-1].set_pose(pos, quat)
        pgs[-1].set_gripper_width(w)
        pgs[-1].change_color(rgb=rgb, a=a)
    [pg.remove_gripper() for pg in pgs]
