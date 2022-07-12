#  Copyright (c) 6.2021. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import argparse
import numpy as np
import random
from external.virtualhome.simulation.unity_simulator import comm_unity
from utils.virtualhome.vhome_utils import get_nodes_in_room, open_doors, remove_objects, generate_cameras, \
    correct_door_bbox, get_nodes_for_det, clean_nodes_in_room, clean_det_objects, class_mapping
from utils.virtualhome import dataset_config, LIMBS, valid_joint_ids
from utils.virtualhome.read_frames import read_frames
from utils.vis_base import VIS_BASE
import seaborn as sns
import vtk
from external.virtualhome.simulation.unity_simulator.utils_viz import get_skeleton
import subprocess
import time

class VIS_HOME(VIS_BASE):
    def __init__(self, nodes=(), cam_locs=(), points_world=(), room_bbox=None, skeleton_joints=None, skip_rates=5):
        super(VIS_HOME, self).__init__()
        self.nodes = nodes
        self.cam_locs = cam_locs
        self.room_bbox = room_bbox
        if len(nodes):
            self.class_ids = class_mapping([node['class_name'] for node in nodes])[0]
            # set palette
            self.palette_cls = np.array([*sns.color_palette("hls", len(dataset_config.class_labels))])
        if isinstance(skeleton_joints, np.ndarray):
            self.move_traj = skeleton_joints[:, 0]
            skeleton_joints = skeleton_joints[::skip_rates]
            self.skeleton_joints = np.zeros([skeleton_joints.shape[0], max(valid_joint_ids) + 1, 3])
            self.skeleton_joints[:, valid_joint_ids] = skeleton_joints
            self.frame_num = skeleton_joints.shape[0]
            self.skeleton_colors = sns.color_palette("Spectral_r", n_colors=self.frame_num)
            self.traj_palette = np.array(sns.color_palette("Spectral_r", n_colors=self.move_traj.shape[0]))
        self.point_cloud = points_world

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''draw 3D boxes'''
        if 'room_bbox' in kwargs['type']:
            # draw room bbox
            centroid = self.room_bbox['centroid']
            vectors = np.diag(np.array(self.room_bbox['size']) / 2.).dot(self.room_bbox['R_mat'])
            color = [125, 125, 125]
            box_actor = self.get_bbox_cube_actor(centroid, vectors, color, 0.05)
            box_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(box_actor)

        if 'bboxes' in kwargs['type']:
            # draw instance bboxes
            for node_idx, node in enumerate(self.nodes):
                centroid = node['centroid']
                vectors = np.diag(np.array(node['size']) / 2.).dot(node['R_mat'])
                color = self.palette_cls[self.class_ids[node_idx]] * 255
                box_actor = self.get_bbox_line_actor(centroid, vectors, color, 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        '''draw camera locations'''
        if 'cam_locs' in kwargs['type']:
            for cam_loc in self.cam_locs:
                sphere_actor = self.set_actor(
                    self.set_mapper(self.set_sphere_property(cam_loc[0], 0.1), mode='model'))
                sphere_actor.GetProperty().SetColor([0.8, 0.1, 0.1])
                sphere_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(sphere_actor)

        '''draw scene mesh'''
        if 'mesh' in kwargs['type']:
            ply_actor = self.set_actor(self.set_mapper(self.set_ply_property('./temp/tsdf_mesh_vhome.ply'), 'model'))
            ply_actor.GetProperty().SetOpacity(1)
            ply_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(ply_actor)

        if 'voxels' in kwargs['type']:
            ply_actor = self.set_actor(self.set_mapper(self.set_ply_property('./temp/voxel_mesh_vhome.ply'), 'model'))
            ply_actor.GetProperty().SetOpacity(1)
            ply_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(ply_actor)

        if 'points' in kwargs['type']:
            '''render points'''
            points = np.vstack(self.point_cloud['pc'])
            colors = np.vstack(self.point_cloud['color'])
            point_size = 3

            point_actor = self.set_actor(self.set_mapper(self.set_points_property(points, colors), 'box'))
            point_actor.GetProperty().SetPointSize(point_size)
            point_actor.GetProperty().SetOpacity(1)
            point_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(point_actor)

        if 'skeleton' in kwargs['type']:
            '''render skeleton joints'''
            for sk_idx, skeleton in enumerate(self.skeleton_joints):
                # draw joints
                for jt_idx, joint in enumerate(skeleton):
                    if jt_idx not in valid_joint_ids:
                        continue
                    if jt_idx == 10:
                        radius = 0.1
                    else:
                        radius = 0.05

                    sphere_actor = self.set_actor(
                        self.set_mapper(self.set_sphere_property(joint, radius), mode='model'))
                    sphere_actor.GetProperty().SetColor(self.skeleton_colors[sk_idx])
                    sphere_actor.GetProperty().SetInterpolationToPBR()
                    renderer.AddActor(sphere_actor)

                # draw lines
                for line_idx, line in enumerate(LIMBS):
                    p0 = skeleton[line[0]]
                    p1 = skeleton[line[1]]
                    line_actor = self.set_actor(self.set_mapper(self.set_line_property(p0, p1), mode='model'))
                    line_actor.GetProperty().SetLineWidth(6)
                    line_actor.GetProperty().SetColor(self.skeleton_colors[sk_idx])
                    line_actor.GetProperty().SetInterpolationToPBR()
                    renderer.AddActor(line_actor)

            # draw directions
            for traj_id in range(self.move_traj.shape[0] - 1):
                if np.linalg.norm(self.move_traj[traj_id + 1] - self.move_traj[traj_id]) == 0.:
                    continue
                line_actor = self.set_actor(
                    self.set_mapper(self.set_line_property(self.move_traj[traj_id], self.move_traj[traj_id + 1]),
                                    mode='model'))
                line_actor.GetProperty().SetLineWidth(5)
                line_actor.GetProperty().SetColor(self.traj_palette[traj_id])
                line_actor.GetProperty().SetOpacity(1)
                line_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(line_actor)

        '''light'''
        focal_point = np.array([0., 0., 0.])
        positions = focal_point + np.array([(100, 100, 100), (100, 100, -100), (-100, 100, 100), (-100, 100, -100)])
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(0.8)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(*focal_point)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Virtual Room Visualization.')
    parser.add_argument('--scene-id', type=int, default=3,
                        help='Give a scene id in [0-7].')
    parser.add_argument('--room-id', type=int, default=0,
                        help='Give a scene id in [0-N].')
    parser.add_argument('--script-id', type=int, default=0,
                        help='Give a script id.')
    parser.add_argument('--char-id', type=int, default=1,
                        help='Give a character id in [0-5].')
    parser.add_argument('--vis_mesh', action='store_true',
                        help='If visualize scene mesh.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(dataset_config.random_seed)
    random.seed(dataset_config.random_seed)
    '''Start Unity and continue'''
    unity_laucher = subprocess.Popen(dataset_config.unity_lauch_cmd)
    time.sleep(5)
    '''Build scene communication'''
    comm = comm_unity.UnityCommunication(timeout_wait=3000)

    '''Reset the scene'''
    comm.reset(args.scene_id)

    '''Get graph'''
    s, graph = comm.environment_graph()
    nodes = graph['nodes']
    edges = graph['edges']
    # check if has redundant nodes
    all_node_ids = [node['id'] for node in nodes]
    assert len(set(all_node_ids)) == len(all_node_ids)

    '''open doors to avoid ambiguous pose animations'''
    nodes = open_doors(nodes)

    '''Get room nodes'''
    room_nodes = []
    for node in nodes:
        if node['category'] == 'Rooms':
            room_nodes.append(node)

    # make sure the room id for visualize not exceed the maximal room count.
    assert args.room_id <= len(room_nodes) + 1
    room_node = room_nodes[args.room_id]

    '''Parse all nodes in this room'''
    nodes_in_room, edges_in_room = get_nodes_in_room(nodes, edges, room_node)

    '''Refine nodes in room, get all nodes in refined room bbox'''
    nodes_in_room, edges_in_room, room_bbox = clean_nodes_in_room(nodes_in_room, edges_in_room, room_node)

    '''update graph'''
    graph_update = {}
    graph_update['nodes'] = nodes
    graph_update['edges'] = edges
    success = comm.expand_scene(graph_update)
    assert success[0]

    '''get the object bbox for detection'''
    nodes_for_det, edges_for_det = remove_objects(nodes_in_room, edges_in_room, dataset_config.class_labels_raw,
                                                  level='class', mode='include')

    '''clean nodes_for_det that are not interactable'''
    nodes_for_det, edges_for_det, grabbale_nodes, interactable_node_cmds = clean_det_objects(comm, args.scene_id,
                                                                                             graph_update, room_node,
                                                                                             nodes_in_room,
                                                                                             edges_in_room,
                                                                                             nodes_for_det,
                                                                                             edges_for_det,
                                                                                             dataset_config)
    comm.reset(args.scene_id)
    success = comm.expand_scene(graph_update)
    assert success[0]

    '''correct bboxes for doors'''
    nodes_for_det = correct_door_bbox(nodes_for_det, nodes_in_room)
    nodes_for_det = get_nodes_for_det(nodes_for_det)

    '''read skeletons'''
    char_name = dataset_config.character_names[args.char_id]
    sk_input_path = dataset_config.recording_path.joinpath(str(args.scene_id), str(args.room_id), str(args.script_id),
                                                           char_name.split('/')[1])
    skeleton_joints, frames_ids = get_skeleton(sk_input_path, 'script')

    # '''visualize bboxes'''
    if not args.vis_mesh:
        viser = VIS_HOME(nodes=nodes_for_det, room_bbox=room_bbox, skeleton_joints=skeleton_joints, skip_rates=10)
        viser.visualize(type=['bboxes', 'room_bbox', 'skeleton'])

        '''Terminate Unity'''
        print('---End---')
        unity_laucher.kill()
        sys.exit(0)

    '''open doors'''
    nodes_in_room = open_doors(nodes_in_room)

    '''Remove walls and ceiling for rendering'''
    nodes_in_room, edges_in_room = remove_objects(nodes_in_room, edges_in_room, dataset_config.category_not_render,
                                                  level='category', mode='exclude')

    '''update graph'''
    graph_update = {}
    graph_update['nodes'] = nodes_in_room
    graph_update['edges'] = edges_in_room
    success = comm.expand_scene(graph_update)
    assert success[0]

    '''generate cameras'''
    s, existing_cam_count = comm.camera_count()
    cam_locs, cam_pitch_yaws = generate_cameras(room_node, room_bbox, nodes_in_room, dataset_config)

    '''add cameras'''
    for cam_loc in cam_locs:
        for pitch_yaw in cam_pitch_yaws:
            comm.add_camera(position=cam_loc[0], rotation=[pitch_yaw[0], pitch_yaw[1], 0.])  # pitch, yaw, roll

    s, current_cam_count = comm.camera_count()

    frame_ids = list(range(existing_cam_count, current_cam_count))

    '''generate mesh/point clouds/bbox data'''
    points_world, voxel_origin = read_frames(comm, frame_ids, dataset_config, if_vis=True)

    '''visualize cam_locs, point_clouds, meshes, etc'''
    viser = VIS_HOME(nodes=nodes_for_det, cam_locs=cam_locs, points_world=points_world, skeleton_joints=skeleton_joints)
    viser.visualize(type=['voxels', 'bboxes', 'skeleton'])

    '''Terminate Unity'''
    unity_laucher.kill()