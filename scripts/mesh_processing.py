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


def get_obj_urdf(name, m=1.0, s=1.0):
    robot = et.Element('robot')
    robot.set('name', name)
    link = et.SubElement(robot, 'link')
    link.set('name', name)
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
    mass.set('value', '{}'.format(m))
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
    mesh.set('filename', name + '_vis.obj')
    mesh.set('scale', '{} {} {}'.format(s, s, s))
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
    mesh.set('filename', name + '_col.obj')
    mesh.set('scale', '{} {} {}'.format(s, s, s))
    xml_str = minidom.parseString(et.tostring(robot)).toprettyxml(indent='  ')
    return xml_str


def dexnet(mesh_path, output):
    ms = ml.MeshSet()
    for obj_name in os.listdir(os.path.join(mesh_path)):
        print(obj_name)
        obj_name = obj_name[:-4]
        obj_path = os.path.join(output, obj_name)
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        ms.load_new_mesh(os.path.join(mesh_path, obj_name + '.obj'))
        ms.apply_filter('meshing_invert_face_orientation', forceflip=False)
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '.obj'))
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_vis.obj'))
        # ms.apply_filter('generate_convex_hull')
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_col.obj'))
        xml_str = get_obj_urdf(obj_name, m=1.0, s=1.0)
        with open(os.path.join(obj_path, obj_name + '.urdf'), 'w') as f:
            f.write(xml_str)


def kit(mesh_path, output):
    ms = ml.MeshSet()
    for obj_name in os.listdir(os.path.join(mesh_path)):
        print(obj_name)
        obj_name = obj_name[:-4]
        obj_path = os.path.join(output, obj_name)
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        ms.load_new_mesh(os.path.join(mesh_path, obj_name + '.obj'))
        ms.apply_filter('meshing_invert_face_orientation', forceflip=False)
        # ms.apply_filter('transform_align_to_principal_axis')
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '.obj'))
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_col.obj'))
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_vis.obj'))
        xml_str = get_obj_urdf(obj_name, m=1.0, s=1.0)
        with open(os.path.join(obj_path, obj_name + '.urdf'), 'w') as f:
            f.write(xml_str)


def block(mesh_path, output):
    ms = ml.MeshSet()
    for obj_name in os.listdir(os.path.join(mesh_path)):
        print(obj_name)
        obj_name = obj_name[:-4]
        obj_path = os.path.join(output, obj_name)
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        ms.load_new_mesh(os.path.join(mesh_path, obj_name + '.obj'))
        ms.apply_filter('meshing_invert_face_orientation', forceflip=False)
        # ms.apply_filter('transform_align_to_principal_axis')
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '.obj'))
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_col.obj'))
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_vis.obj'))
        xml_str = get_obj_urdf(obj_name, m=1.0, s=1.0)
        with open(os.path.join(obj_path, obj_name + '.urdf'), 'w') as f:
            f.write(xml_str)


def egad(mesh_path, output):
    ms = ml.MeshSet()
    for obj_name in os.listdir(os.path.join(mesh_path)):
        print(obj_name)
        obj_name = obj_name[:-4]
        obj_path = os.path.join(output, obj_name)
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        ms.load_new_mesh(os.path.join(mesh_path, obj_name + '.obj'))
        # ms.apply_filter('transform_align_to_principal_axis')
        ms.apply_filter('transform_translate_center_set_origin', traslmethod=2)
        ms.apply_filter('transform_scale_normalize', axisx=0.1)
        ms.apply_filter('transform_scale_normalize', axisx=0.1)
        ms.apply_filter('transform_scale_normalize', axisx=0.1)
        min_size = np.min(ms.current_mesh().bounding_box().max() - ms.current_mesh().bounding_box().min())
        scale_factor = max_width * 0.8 / min_size
        if scale_factor < 1:
            ms.apply_filter('transform_scale_normalize', axisx=scale_factor)
        ms.apply_filter('repair_non_manifold_edges')
        # ms.apply_filter('subdivision_surfaces_midpoint')
        # ms.apply_filter('taubin_smooth', lambda_=1.0)
        ms.apply_filter('taubin_smooth', lambda_=0.5)
        ms.apply_filter('laplacian_smooth_surface_preserving', angledeg=0.5)
        # ms.apply_filter('simplification_clustering_decimation')
        ms.apply_filter('re_compute_face_normals')
        ms.apply_filter('normalize_face_normals')
        ms.apply_filter('re_compute_vertex_normals', weightmode=2)
        ms.apply_filter('normalize_vertex_normals')
        ms.apply_filter('remove_duplicate_vertices')
        ms.apply_filter('remove_duplicate_faces')
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '.obj'))
        ms.apply_filter('simplification_quadric_edge_collapse_decimation')
        ms.apply_filter('simplification_quadric_edge_collapse_decimation')
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_col.obj'))
        ms.save_current_mesh(os.path.join(obj_path, obj_name + '_vis.obj'))
        gm = ms.apply_filter('compute_geometric_measures')
        bbox_volume = ms.current_mesh().bounding_box().dim_x() * ms.current_mesh().bounding_box().dim_y() * ms.current_mesh().bounding_box().dim_z()
        m = bbox_volume * 137
        xml_str = get_obj_urdf(obj_name, m)
        with open(os.path.join(obj_path, obj_name+'.urdf'), 'w') as f:
            f.write(xml_str)


def primitive(mesh_path, output):
    with open('config/primitives.json', 'r') as f:
        shape_cfg = json.load(f)
    for shape_type in shape_cfg.keys():
        curr_shape = shape_cfg[shape_type]
        shape_type = shape_type.split('#')[0]
        for stride in range(curr_shape['num_stride']):
            scale_x = curr_shape['x'] + curr_shape['stride_x'] * stride
            scale_y = curr_shape['y'] + curr_shape['stride_y'] * stride
            scale_z = curr_shape['z'] + curr_shape['stride_z'] * stride
            folder = shape_type + '#{:.3f}#{:.3f}#{:.3f}'.format(scale_x, scale_y, scale_z)
            output_path = os.path.join(output, folder)
            if os.path.exists(output_path) and len(os.listdir(output_path)) != 0:
                print(folder + ' exists.')
                continue
            else:
                os.makedirs(output_path)
            mesh_path = os.path.join(mesh_path, shape_type)
            for mesh_name in os.listdir(mesh_path):
                mesh = trimesh.load_mesh(os.path.join(mesh_path, mesh_name))
                mesh.apply_scale((scale_x, scale_y, scale_z))
                mesh.visual = trimesh.visual.ColorVisuals()
                strings = mesh_name.split('_')
                mesh_name = '_'.join([folder, strings[-1]]) if len(strings) == 2 else folder + '.obj'
                mesh.export(os.path.join(output_path, mesh_name))
            xml_str = get_obj_urdf(mesh_name)
            with open(os.path.join(output_path, folder + '.urdf'), 'w') as f:
                f.write(xml_str)


def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    globals()[args.mesh_type](args.mesh_path, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mesh processing on EGAD dataset.')
    parser.add_argument('--mesh_path',
                        type=str,
                        required=True,
                        help='path of the mesh set.')
    parser.add_argument('--mesh_type',
                        type=str,
                        required=True,
                        help='type of the mesh set.')
    parser.add_argument('--w',
                        type=float,
                        default=0.08,
                        help='the maximal width of the gripper.')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='path to save the processed mesh set.')
    main(parser.parse_args())
