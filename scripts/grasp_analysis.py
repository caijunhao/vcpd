from sim.utils import basic_rot_mat
from sim.objects import RigidObject, PandaGripper
from scipy.spatial.transform import Rotation
import argparse
import pybullet as p
import numpy as np
import trimesh
import time
import shutil
import json
import os


def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    mode = p.GUI if args.gui else p.DIRECT
    physics_id = p.connect(mode)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -9.8)
    with open(args.config, 'r') as config_file:
        cfg = json.load(config_file)
    pg = PandaGripper('assets')
    pg.set_pose([1, 0, 0], [0, 0, 0, 1])
    obj_list = os.listdir(os.path.join(args.mesh_path))
    np.random.shuffle(obj_list)
    for obj_name in obj_list:
        print(obj_name)
        obj_path = os.path.join(args.mesh_path, obj_name, obj_name+'.obj')
        mesh = trimesh.load(obj_path)
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': obj_path, 'meshScale': [1]*3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': obj_path, 'meshScale': [1]*3}
        body_params = {'baseMass': 0, 'basePosition': [0, 0, 0], 'baseOrientation': [0, 0, 0, 1]}
        obj = RigidObject(obj_name, vis_params=vis_params, col_params=col_params, body_params=body_params)
        mean_edge_distance = np.mean(np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=1))
        num_vertices = mesh.vertices.shape[0]
        # vertex index, direction, intersect face index, intersect vertex, center, antipodal raw, mean, min
        grasp_info = {'vertex_ids': [np.zeros((0,))],
                      'directions': [np.zeros((0, 3))],
                      'intersects': [np.zeros((0, 3))],
                      'intersected_face_ids': [np.zeros((0,))],
                      'centers': [np.zeros((0, 3))],
                      'widths': [np.zeros((0,))],
                      'antipodal_raw': [np.zeros((0,))],
                      'antipodal_mean': [np.zeros((0,))],
                      'antipodal_min': [np.zeros((0,))],
                      'collisions': [np.zeros((0, 64))],
                      'quaternions': [np.zeros((0, 4))]}
        for i in range(num_vertices):
            vertex, normal = mesh.vertices[i], mesh.vertex_normals[i]
            direction = np.mean(mesh.face_normals[mesh.vertex_faces[i]], axis=0)
            direction = direction / np.linalg.norm(direction)
            normal = normal / np.linalg.norm(normal)
            pa, pb = vertex + normal * 0.2, vertex - normal * 0.2
            line_id = p.addUserDebugLine(pa, pb,
                                         lineColorRGB=[0, 0, 1], lineWidth=0.1, lifeTime=0, physicsClientId=physics_id)
            pa, pb = vertex + direction * 0.2, vertex - direction * 0.2
            line2 = p.addUserDebugLine(pa, pb,
                                       lineColorRGB=[0, 1, 0], lineWidth=0.1, lifeTime=0, physicsClientId=physics_id)
            vij = mesh.vertices - vertex.reshape(1, 3)  # vectors from i to j
            dist = np.linalg.norm(
                vij - np.sum(vij * normal.reshape(1, 3), axis=1, keepdims=True) * normal.reshape(1, 3), axis=1)
            dist[i] = 1e10
            dist[np.dot(mesh.vertex_normals, normal) >= 0] = 1e10
            flag = dist <= 2 * mean_edge_distance
            vertex_faces = mesh.vertex_faces[flag]
            vertex_ids = mesh.faces[vertex_faces]
            # visualize vertices
            # spheres = []
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            # for vertex_idx in vertex_ids.reshape(-1):
            #     spheres.append(p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, 0.001),
            #                                      p.createVisualShape(p.GEOM_SPHERE, 0.001),
            #                                      basePosition=mesh.vertices[vertex_idx]))
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            # [p.removeBody(sphere_id) for sphere_id in spheres]
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            # shape of vertex_coords: num_candidates * max_adjacent_faces * num_vertices_per_triangle * num_coords
            vertex_coords = mesh.vertices[vertex_ids]  # triangle vertices
            # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
            p0, p1, p2 = [np.squeeze(arr, axis=2) for arr in np.split(vertex_coords, 3, axis=2)]
            la = vertex.reshape(1, 1, 3)
            lab = normal.reshape(1, 1, 3)
            p01 = p1 - p0
            p02 = p2 - p0
            denominator = np.sum(-lab * np.cross(p01, p02), axis=2)
            denominator[denominator == 0] = 1e-7
            t = np.sum(np.cross(p01, -p02) * (la - p0), axis=2) / denominator
            u = np.sum(np.cross(p02, -lab) * (la - p0), axis=2) / denominator
            v = np.sum(np.cross(-lab, p01) * (la - p0), axis=2) / denominator
            intersects = la - lab * np.expand_dims(t, axis=-1)
            intersect_flag = (u >= 0) * (u <= 1) * (v >= 0) * (v <= 1) * ((u + v) >= 0) * ((u + v) <= 1)
            if np.any(intersect_flag):
                selected_faces, selected_ids = np.unique(vertex_faces[intersect_flag], return_index=True)
                selected_intersects = intersects[intersect_flag][selected_ids]
                selected_ids = np.unique(np.sum(selected_intersects*1e4, axis=1).astype(int), return_index=True)[1]
                selected_faces, selected_intersects = selected_faces[selected_ids], selected_intersects[selected_ids]
                num_intersects = selected_intersects.shape[0]
                widths = np.linalg.norm((vertex.reshape(1, 3) - selected_intersects), axis=1)
                # width_flag = widths < max_width
                # selected_faces = selected_faces[width_flag]
                # selected_intersects = selected_intersects[width_flag]
                # widths = widths[width_flag]
                centers = (vertex.reshape(1, 3) + selected_intersects) / 2
                face_normals = mesh.face_normals[selected_faces]
                face_vertex_normals = mesh.vertex_normals[mesh.faces[selected_faces]]
                cos1s = np.abs(np.sum(face_vertex_normals * normal.reshape(1, 1, 3), axis=-1))
                vertex_face_ids = mesh.vertex_faces[i][mesh.vertex_faces[i] != -1]
                vertex_face_normals = mesh.face_normals[vertex_face_ids]
                cos2s = np.abs(np.dot(vertex_face_normals, normal))
                mean_cos1, min_cos1 = np.mean(cos1s, axis=1), np.min(cos1s, axis=1)
                mean_cos2, min_cos2 = np.mean(cos2s), np.min(cos2s)
                mean_score, min_score = mean_cos1 * mean_cos2, min_cos1 * min_cos2
                raw = np.abs(np.dot(face_normals, normal))
                # print('original antipodal score: {}'.format(raw))
                # print('mean cos1: {} | mean cos2: {} | mean antipodal score: {}'.format(mean_cos1, mean_cos2, mean_score))
                # print('min cos1: {} | min cos2: {} | min antipodal score: {}'.format(min_cos1, min_cos2, min_score))
                # vertex index, direction, intersect face index, intersect vertex, center, antipodal raw, mean, min
                curr_idx = np.array([i] * num_intersects)
                directions = np.stack([normal] * num_intersects, axis=0)
                cols = np.ones((num_intersects, cfg['num_angle']))
                quats = np.zeros((num_intersects, 4))
                quats[..., -1] = 1
                for j in range(num_intersects):
                    if min_score[j] >= cfg['th_min']:
                        y = directions[j]
                        mat, _, _ = np.linalg.svd(y.reshape(3, 1))
                        x = mat[:, 1]
                        z = np.cross(x, y)
                        base = np.stack([x, y, z], axis=1)
                        angles = np.arange(cfg['num_angle']) / cfg['num_angle'] * np.pi * 2
                        delta_rots = basic_rot_mat(angles, axis='y')
                        rots = np.matmul(base.reshape((1, 3, 3)), delta_rots)
                        quat = Rotation.from_matrix(rots).as_quat()
                        if np.sum(np.sum(np.abs(quat), axis=1) == 0) > 0:
                            print(1)
                            pass
                        for angle_idx in range(cfg['num_angle']):
                            pos = centers[j].copy() - rots[angle_idx, :, 2] * cfg['gripper']['depth']
                            pg.set_pose(pos, quat[angle_idx])
                            pg.set_gripper_width(widths[j] + 0.02)
                            cols[j, angle_idx] = int(pg.is_collided([]))
                        quats[j] = quat[0]
                grasp_info['vertex_ids'].append(curr_idx)
                grasp_info['directions'].append(directions)
                grasp_info['intersects'].append(selected_intersects)
                grasp_info['intersected_face_ids'].append(selected_faces)
                grasp_info['centers'].append(centers)
                grasp_info['widths'].append(widths)
                grasp_info['antipodal_raw'].append(raw)
                grasp_info['antipodal_mean'].append(mean_score)
                grasp_info['antipodal_min'].append(min_score)
                grasp_info['collisions'].append(cols)
                grasp_info['quaternions'].append(quats)
            else:
                grasp_info['vertex_ids'].append(np.array([i]))
                grasp_info['directions'].append(np.zeros((1, 3)))
                grasp_info['intersects'].append(np.zeros((1, 3)))
                grasp_info['intersected_face_ids'].append(np.array([0]))
                grasp_info['centers'].append(np.zeros((1, 3)))
                grasp_info['widths'].append(np.array([0.08]))
                grasp_info['antipodal_raw'].append(np.array([0]))
                grasp_info['antipodal_mean'].append(np.array([0]))
                grasp_info['antipodal_min'].append(np.array([0]))
                grasp_info['collisions'].append(np.ones((1, 64)))
                grasp_info['quaternions'].append(Rotation.from_matrix(np.eye(3)).as_quat().reshape(1, 4))
            p.removeUserDebugItem(line_id, physics_id)
            p.removeUserDebugItem(line2, physics_id)
        for k, v in grasp_info.items():
            grasp_info[k] = np.concatenate(v, axis=0)
        num_pairs = grasp_info['vertex_ids'].shape[0]
        num_graspable_pairs = np.sum(grasp_info['antipodal_min'] >= cfg['th_min'])
        num_col_free_poses = np.sum(grasp_info['collisions'] == 0)
        print('# pairs: {}'.format(num_pairs))
        print('# graspable pairs: {}'.format(num_graspable_pairs))
        print('# poses: {}'.format(num_pairs * cfg['num_angle']))
        print('# col-free poses: {}'.format(num_col_free_poses))
        print('graspable ratio: {}'.format(num_col_free_poses / (num_pairs * cfg['num_angle'])))
        if num_col_free_poses == 0:
            print('no graspable pair found, {} will be removed.'.format(obj_name))
            shutil.rmtree(os.path.join(args.mesh_path, obj_name))
            continue
        with open(os.path.join(args.output, '{}_info.json'.format(obj_name)), 'w') as f:
            d = {'num_pairs': int(num_pairs),
                 'num_graspable_pairs': int(num_graspable_pairs),
                 'num_poses': int(num_pairs * cfg['num_angle']),
                 'num_col-free_poses': int(num_col_free_poses),
                 'graspable_ratio': float(num_col_free_poses / (num_pairs * cfg['num_angle'])),
                 'keys': list(grasp_info.keys())}
            json.dump(d, f, indent=4)
        for k, v in grasp_info.items():
            np.save(os.path.join(args.output, '{}_{}.npy'.format(obj_name, k)), v)
        p.removeBody(obj.obj_id)
        del obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='antipodal grasp analysis on single mesh.')
    parser.add_argument('--mesh_path',
                        type=str,
                        required=True,
                        help='path of the mesh set.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path of the configuration file.')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='path to save the grasp labels.')
    parser.add_argument('--gui',
                        type=int,
                        default=0,
                        help='choose 0 for DIRECT mode and 1 (or others) for GUI mode.')
    main(parser.parse_args())

