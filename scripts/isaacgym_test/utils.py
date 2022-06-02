import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
def vpn_predict(tsdf, vpn, dg, rn, gpr_pts, device):
    from vpn.utils import rank_and_group_poses
    from src.sim.utils import basic_rot_mat

    v, _, n, _ = tsdf.compute_mesh(step_size=2)
    v, n = torch.from_numpy(v).to(device), torch.from_numpy(n).to(device)
    ids = tsdf.get_ids(v)
    sample = dict()
    sample['sdf_volume'] = tsdf.gaussian_blur(tsdf.post_processed_volume).unsqueeze(dim=0).unsqueeze(dim=0)
    sample['pts'] = v.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['normals'] = n.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['ids'] = ids.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['occupancy_volume'] = torch.zeros_like(sample['sdf_volume'], device=device)
    sample['occupancy_volume'][sample['sdf_volume'] <= 0] = 1
    sample['origin'] = tsdf.origin
    sample['resolution'] = torch.tensor([[tsdf.res]], dtype=torch.float32, device=device)
    out = torch.sigmoid(vpn.forward(sample))
    groups = rank_and_group_poses(sample, out, device, gpr_pts, collision_check=True)
    if groups is None:
        return None, None, None

    score, pose = groups[0]['queue'].get()
    # pos, rot0 = pose[0:3], R.from_quat(pose[6:10]).as_matrix()
    # rot = rot0 @ basic_rot_mat(np.pi / 2, axis='z').astype(np.float32)
    # quat = R.from_matrix(rot).as_quat()
    # gripper_pos = pos - 0.08 * rot[:, 2]

    for _ in range(5):
        trans = (pose[0:3] + pose[3:6] * 0.01).astype(np.float32)
        rot = R.from_quat(pose[6:10]).as_matrix().astype(np.float32)
        pos = dg.sample_grippers(trans.reshape(1, 3), rot.reshape(1, 3, 3), inner_only=False)
        pos = np.expand_dims(pos.transpose((2, 1, 0)), axis=0)  # 1 * 3 * 2048 * 1
        sample['perturbed_pos'] = torch.from_numpy(pos).to(device)
        delta_rot = rn(sample)
        delta_rot = torch.squeeze(delta_rot).detach().cpu().numpy()
        rot_recover = rot @ delta_rot
        approach_recover = rot_recover[:, 2]
        quat_recover = R.from_matrix(rot_recover).as_quat()
        trans_recover = trans
        # trans_recover[2] -= 0.005
        pos_recover = pose[0:3]
        pose = np.concatenate([pos_recover, -approach_recover, quat_recover])
    pos, rot0 = pose[0:3], R.from_quat(pose[6:10]).as_matrix()
    rot = rot0 @ basic_rot_mat(np.pi / 2, axis='z').astype(np.float32)
    quat = R.from_matrix(rot).as_quat()
    gripper_pos = pos - 0.08 * rot[:, 2]
    return gripper_pos, rot, quat

def cpn_predict(tsdf, cpn, pg, cfg):
    from cpn.utils import sample_contact_points, select_gripper_pose, clustering
    cp1s, cp2s = sample_contact_points(tsdf, post_processed=True, gaussian_blur=True, start=0.01,
                                       step_size=2)
    ids_cp1, ids_cp2 = tsdf.get_ids(cp1s), tsdf.get_ids(cp2s)
    sample = dict()
    sample['sdf_volume'] = tsdf.gaussian_blur(tsdf.post_processed_volume).unsqueeze(dim=0).unsqueeze(dim=0)
    sample['ids_cp1'] = ids_cp1.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['ids_cp2'] = ids_cp2.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    out = torch.squeeze(cpn.forward(sample))
    gripper_poses, rots, widths, cp1s_, cp2s_ = select_gripper_pose(tsdf, pg.vertex_sets,
                                                                    out, cp1s, cp2s, cfg['gripper']['depth'],
                                                                    check_tray=True,
                                                                    th_s=0.997,
                                                                    th_c=0.999)

    if gripper_poses is None:
        return None, None, None, None, None
    pos, rot, width, cp1, cp2 = clustering(gripper_poses, rots, widths, cp1s_, cp2s_)
    quat = R.from_matrix(rot).as_quat()

    return pos, rot, quat, cp1, cp2
