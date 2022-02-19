from functools import reduce
from xml.etree import ElementTree as et
from xml.dom import minidom
import numpy as np
import pymeshlab as ml
import trimesh
import argparse
import json
import os


max_width = 0.08


def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    ms = ml.MeshSet()
    with open(args.shape_cfg, 'r') as f:
        shape_cfg = json.load(f)
    for shape_type in shape_cfg.keys():
        curr_shape = shape_cfg[shape_type]
        shape_type = shape_type.split('#')[0]
        for stride in range(curr_shape['num_stride']):
            scale_x = curr_shape['x'] + curr_shape['stride_x'] * stride
            scale_y = curr_shape['y'] + curr_shape['stride_y'] * stride
            scale_z = curr_shape['z'] + curr_shape['stride_z'] * stride
            folder = shape_type + '#{:.3f}#{:.3f}#{:.3f}'.format(scale_x, scale_y, scale_z)
            output_path = os.path.join(args.output, folder)
            if os.path.exists(output_path) and len(os.listdir(output_path)) != 0:
                print(folder + ' exists.')
                continue
            else:
                os.makedirs(output_path)
            mesh_path = os.path.join(args.mesh_path, shape_type)
            for mesh_name in os.listdir(mesh_path):
                mesh = trimesh.load_mesh(os.path.join(mesh_path, mesh_name))
                mesh.apply_scale((scale_x, scale_y, scale_z))
                mesh.visual = trimesh.visual.ColorVisuals()
                strings = mesh_name.split('_')
                mesh_name = '_'.join([folder, strings[-1]]) if len(strings) == 2 else folder+'.obj'
                mesh.export(os.path.join(output_path, mesh_name))
            robot = et.Element('robot')
            robot.set('name', folder)
            link = et.SubElement(robot, 'link')
            link.set('name', folder)
            contact = et.SubElement(link, 'contact')
            lateral_friction = et.SubElement(contact, 'lateral_friction')
            lateral_friction.set('value', '1.0')
            rolling_friction = et.SubElement(contact, 'rolling_friction')
            rolling_friction.set('value', '1.0')
            inertia_scaling = et.SubElement(contact, 'inertia_scaling')
            inertia_scaling.set('value', '3.0')
            contact_cdm = et.SubElement(contact, 'contact_cdm')
            contact_cdm.set('value', '0.0')
            contact_erp = et.SubElement(contact, 'contact_erp')
            contact_erp.set('value', '1.0')
            inertial = et.SubElement(link, 'inertial')
            origin = et.SubElement(inertial, 'origin')
            origin.set('rpy', '0 0 0')
            origin.set('xyz', '0 0 0')
            mass = et.SubElement(inertial, 'mass')
            mass.set('value', '{}'.format(1))
            inertia = et.SubElement(inertial, 'inertia')
            inertia.set('ixx', '0')
            inertia.set('ixy', '0')
            inertia.set('ixz', '0')
            inertia.set('iyy', '0')
            inertia.set('iyz', '0')
            inertia.set('izz', '0')
            visual = et.SubElement(link, 'visual')
            origin = et.SubElement(visual, 'origin')
            origin.set('rpy', '0 0 0')
            origin.set('xyz', '0 0 0')
            geometry = et.SubElement(visual, 'geometry')
            mesh = et.SubElement(geometry, 'mesh')
            mesh.set('filename', folder+'_vis.obj')
            mesh.set('scale', '1.0 1.0 1.0')
            material = et.SubElement(visual, 'material')
            material.set('name', 'blockmat')
            color = et.SubElement(material, 'color')
            color.set('rgba', '1.0 1.0 1.0 1.0')
            collision = et.SubElement(link, 'collision')
            origin = et.SubElement(collision, 'origin')
            origin.set('rpy', '0 0 0')
            origin.set('xyz', '0 0 0')
            geometry = et.SubElement(collision, 'geometry')
            mesh = et.SubElement(geometry, 'mesh')
            mesh.set('filename', folder+'_col.obj')
            mesh.set('scale', '1.0 1.0 1.0')

            xml_str = minidom.parseString(et.tostring(robot)).toprettyxml(indent='  ')
            with open(os.path.join(output_path, folder+'.urdf'), 'w') as f:
                f.write(xml_str)


def compute_patch_normal(mesh, orders_of_neigh=1):
    """
    Compute patch normal for each vertex of the given mesh by principle component analysis
    :param mesh: trimesh.Trimesh instance. Mesh to exported
    :param orders_of_neigh: The number of orders of neighbors to be considered to compute the patch normal
    :return: A trimesh.Trimesh instance of the new mesh with patch normals as the vertices normals.
    """
    num_vertices = mesh.vertices.shape[0]
    order_list = [np.arange(num_vertices).reshape(num_vertices, 1).tolist()]
    neighbor_ids_list = []
    max_degree = 0
    degrees = np.zeros(num_vertices, dtype=int)
    for i in range(orders_of_neigh):
        if i == 0:
            order_list.append(mesh.vertex_neighbors)
            continue
        curr_neighbors = []
        for j in range(num_vertices):
            vertex_neighbors = [mesh.vertex_neighbors[idx] for idx in order_list[-1][j]]
            vertex_neighbors = reduce(lambda x, y: x+y, vertex_neighbors)
            curr_neighbors.append(list(set(list(vertex_neighbors))))
        order_list.append(curr_neighbors)
    for j in range(num_vertices):
        vertex_neighbors = [order_list[idx][j] for idx in range(len(order_list))]
        vertex_neighbors = reduce(lambda x, y: x+y, vertex_neighbors)
        vertex_neighbors = list(set(list(vertex_neighbors)))
        degrees[j] = len(vertex_neighbors)
        max_degree = max(degrees[j], max_degree)
        neighbor_ids_list.append(vertex_neighbors)
    neighbor_ids_list = [neighbor_ids_list[i]+[-1]*(max_degree-degrees[i]) for i in range(num_vertices)]
    neighbor_ids = np.array(neighbor_ids_list)
    neighbors = mesh.vertices[neighbor_ids]
    neighbors[neighbor_ids == -1] = 0
    mean = np.sum(neighbors, axis=1, keepdims=True) / degrees.reshape((-1, 1, 1))
    neighbors[neighbor_ids == -1] = np.repeat(mean, max_degree, axis=1)[neighbor_ids == -1]
    cov = np.matmul((neighbors - mean).transpose(0, 2, 1), neighbors - mean) / degrees.reshape((-1, 1, 1))
    s, u = np.linalg.eig(cov)
    n_pca = u[..., 0]
    cos = np.sum(n_pca * mesh.vertex_normals, axis=1)
    n_pca[cos < 0] = -n_pca[cos < 0]
    # cos = np.abs(cos)
    new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_normals=n_pca)
    # new_mesh.vertex_normals = n_pca
    return new_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mesh processing on EGAD dataset.')
    parser.add_argument('--mesh_path',
                        type=str,
                        required=True,
                        help='path of the mesh set.')
    parser.add_argument('--shape_cfg',
                        type=str,
                        required=True,
                        help='path to the shape config file.')
    parser.add_argument('--w',
                        type=float,
                        default=0.08,
                        help='the maximal width of the gripper.')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='path to save the processed mesh set.')
    main(parser.parse_args())
