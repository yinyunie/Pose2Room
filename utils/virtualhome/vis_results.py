#  Copyright (c) 9.2021. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import argparse
from utils.virtualhome import dataset_config, LIMBS, valid_joint_ids
from pathlib import Path
import h5py
import numpy as np
from utils.pc_utils import head2rot
from utils.vis_base import VIS_BASE
import vtk
import seaborn as sns
from utils.tools import read_json
import trimesh
from utils.virtualhome.vis_gt_vh import dist_node2bbox

def get_even_dist_joints(skeleton_joints, skip_rates):
    # Downsampling by 1-d distance interpolation
    frame_num = skeleton_joints.shape[0] // skip_rates + 1
    movement_dist = np.linalg.norm(np.diff(skeleton_joints[:, 0], axis=0), axis=1)
    cum_dist = np.cumsum(np.hstack([[0], movement_dist]))
    target_cum_dist = np.linspace(0, sum(movement_dist), frame_num)
    selected_sk_ids = np.argmin(np.abs(cum_dist[:, np.newaxis] - target_cum_dist), axis=0)
    return selected_sk_ids

class VIS_Compare(VIS_BASE):
    def __init__(self, gt_nodes=(), pred_nodes=(), cam_locs=(), points_world=(), room_bbox=None, skeleton_joints=None, skip_rates=1, scene_geo='', keep_interact_skeleton=False, selected_sk_ids=()):
        super(VIS_Compare, self).__init__()
        self.gt_nodes = gt_nodes
        self.pred_nodes = pred_nodes
        self.cam_locs = cam_locs
        self.room_bbox = room_bbox
        self.scene_geo = scene_geo
        self._cam_K = np.array([[1600, 0, 800], [0, 1600, 600], [0, 0, 1]])
        self.gt_class_ids, self.gt_palette_cls = self.get_cls_palatte(gt_nodes)
        self.pred_class_ids, self.pred_palette_cls = self.get_cls_palatte(pred_nodes)
        if isinstance(skeleton_joints, np.ndarray):
            self.move_traj = skeleton_joints[:, 0]
            selected_sk_ids = range(skeleton_joints.shape[0])
            if skip_rates > 1 and not keep_interact_skeleton:
                selected_sk_ids = get_even_dist_joints(skeleton_joints, skip_rates)
                skeleton_joints = skeleton_joints[selected_sk_ids]
            elif keep_interact_skeleton:
                joint_coordinates = skeleton_joints.reshape(-1, 3)
                # get distance between joint to nodes
                selected_sk_ids = dist_node2bbox(self.gt_nodes, joint_coordinates, dataset_config.joint_num)
                # add more frames close to it.
                if skip_rates == 1:
                    local_sk_ids = np.arange(-50, 50, skip_rates)[np.newaxis]
                    selected_sk_ids = selected_sk_ids[:, np.newaxis] + local_sk_ids
                    selected_sk_ids = selected_sk_ids.flatten()
                    selected_sk_ids = selected_sk_ids[selected_sk_ids < skeleton_joints.shape[0]]
                    selected_sk_ids = np.sort(selected_sk_ids)
                else:
                    local_sk_ids = np.arange(-50, 50)[np.newaxis]
                    piece_sk_ids = selected_sk_ids[:, np.newaxis] + local_sk_ids
                    even_dist_sk_ids = [selected_sk_ids]
                    for per_piece_sk_ids in piece_sk_ids:
                        per_piece_sk_ids = per_piece_sk_ids[per_piece_sk_ids < skeleton_joints.shape[0]]
                        picked_ids = get_even_dist_joints(skeleton_joints[per_piece_sk_ids], skip_rates)
                        even_dist_sk_ids.append(per_piece_sk_ids[picked_ids])
                    selected_sk_ids = np.sort(np.hstack(even_dist_sk_ids))
                skeleton_joints = skeleton_joints[selected_sk_ids]
            self.skeleton_joints = np.zeros([skeleton_joints.shape[0], max(valid_joint_ids) + 1, 3])
            self.skeleton_joints[:, valid_joint_ids] = skeleton_joints
            self.frame_num = skeleton_joints.shape[0]
            self.traj_palette = np.array(sns.color_palette("Spectral_r", n_colors=self.move_traj.shape[0]))
            self.skeleton_colors = self.traj_palette[selected_sk_ids]
        self.point_cloud = points_world

    def get_cls_palatte(self, nodes):
        if len(nodes):
            class_ids = [node['class_id'][0] for node in nodes]
            # set palette
            palette_cls = np.array([*sns.color_palette("hls", len(dataset_config.class_labels))])
            return class_ids, palette_cls
        else:
            return None, None

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw world system'''
        # renderer.AddActor(self.set_axes_actor())

        cam_fp = (self.move_traj.max(0) + self.move_traj.min(0))/2.
        cam_loc = cam_fp + kwargs.get('cam_centroid', [3, 10, 3])
        cam_up = [0, sum((cam_loc - cam_fp) ** 2) / (cam_loc[1] - cam_fp[1]), 0] + cam_fp - cam_loc
        camera = self.set_camera(cam_loc, cam_fp, cam_up, self.cam_K)
        renderer.SetActiveCamera(camera)

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
            vis_nodes = self.gt_nodes if kwargs['mode'] == 'gt' else self.pred_nodes
            class_ids = self.gt_class_ids if kwargs['mode'] == 'gt' else self.pred_class_ids
            palette_cls = self.gt_palette_cls if kwargs['mode'] == 'gt' else self.pred_palette_cls
            # draw instance bboxes
            for node_idx, node in enumerate(vis_nodes):
                centroid = node['centroid']
                vectors = np.diag(np.array(node['size']) / 2.).dot(node['R_mat'])
                color = palette_cls[class_ids[node_idx]] * 255
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

        '''draw scene geo'''
        if 'scene_geo' in kwargs['type']:
            ply_actor = self.set_actor(self.set_mapper(self.set_ply_property(self.scene_geo), 'model'))
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
                opacity = 1
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
                    sphere_actor.GetProperty().SetOpacity(opacity)
                    sphere_actor.GetProperty().SetInterpolationToPBR()
                    renderer.AddActor(sphere_actor)

                # draw lines
                for line_idx, line in enumerate(LIMBS):
                    p0 = skeleton[line[0]]
                    p1 = skeleton[line[1]]
                    line_actor = self.set_actor(self.set_mapper(self.set_line_property(p0, p1), mode='model'))
                    line_actor.GetProperty().SetLineWidth(6)
                    line_actor.GetProperty().SetColor(self.skeleton_colors[sk_idx])
                    line_actor.GetProperty().SetOpacity(opacity)
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

def read_gt(sample_filename):
    sample_file = dataset_config.sample_path.joinpath(sample_filename + '.hdf5')
    sample_data = h5py.File(sample_file, "r")
    room_bbox = {}
    for key in sample_data['room_bbox'].keys():
        room_bbox[key] = sample_data['room_bbox'][key][:]
    skeleton_joints = sample_data['skeleton_joints'][:]

    object_nodes = []
    for idx in range(len(sample_data['object_nodes'])):
        object_node = {}
        node_data = sample_data['object_nodes'][str(idx)]
        for key in node_data.keys():
            if node_data[key].shape is None:
                continue
            object_node[key] = node_data[key][:]
        object_nodes.append(object_node)

    return object_nodes, room_bbox, skeleton_joints

def read_pred(sample_filename):
    sample_file = list(pred_path.rglob('*%s*' % (sample_filename)))[0]
    bbox_info = np.load(sample_file.joinpath('000000_pred_confident_nms_bbox.npz'))
    object_nodes = []
    for bbox, cls_label in zip(bbox_info['obbs'], bbox_info['cls']):
        centroid = bbox[:3]
        box_size = bbox[3:6]
        heading_angle = bbox[6]
        R_mat = head2rot(heading_angle)

        object_node = {}
        object_node['centroid'] = centroid
        object_node['R_mat'] = R_mat
        object_node['size'] = box_size
        object_node['class_id'] = [cls_label]
        object_nodes.append(object_node)
    return object_nodes

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Virtual Room Visualization.')
    parser.add_argument('pred_or_gt', type=str,
                        help='Specify to visualize the prediction (pred) or ground-truth (gt).')
    parser.add_argument('--scene-id', type=int, default=3,
                        help='Give a scene id in [0-7].')
    parser.add_argument('--room-id', type=int, default=0,
                        help='Give a scene id in [0-N].')
    parser.add_argument('--script-id', type=int, default=746,
                        help='Give a script id.')
    parser.add_argument('--char-id', type=int, default=1,
                        help='Give a character id in [0-5].')
    parser.add_argument('--pred-path', type=str, default='out/p2rnet/test/2022-07-11T19:05:29.574802/visualization',
                        help='Give the visualiazation path.')
    parser.add_argument('--vis_scene_geo', action='store_true',
                        help='Visualize scene geometry.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    char_name = dataset_config.character_names[args.char_id].split('/')[1]
    sample_filename = '%d_%d_%d_%s_0' % (args.scene_id, args.room_id, args.script_id, char_name)

    '''Read pred data'''
    pred_path = Path(args.pred_path)
    pred_object_nodes = read_pred(sample_filename)

    '''Read GT data'''
    gt_object_nodes, gt_room_bbox, skeleton_joints = read_gt(sample_filename)
    # get room centroid offset
    room_bbox_file = dataset_config.script_bbox_path.joinpath(str(args.scene_id), 'room_bbox_%d.json' % (args.room_id))
    original_room_centroid = np.array(read_json(room_bbox_file)['room_bbox']['centroid'])
    room_centroid_offset = gt_room_bbox['centroid'] - original_room_centroid

    '''visualize bboxes'''
    save_screen_shot_dir = pred_path.parent.joinpath('screenshots', sample_filename)
    if not save_screen_shot_dir.is_dir():
        save_screen_shot_dir.mkdir(parents=True)

    '''Read mesh/voxel grids'''
    vis_args = ['bboxes', 'room_bbox', 'skeleton']
    if args.vis_scene_geo:
        scene_geo = dataset_config.scene_geo_path.joinpath('%s_%s/tsdf_mesh_vhome.ply' % (args.scene_id, args.room_id))
        scene_geo_centered = dataset_config.scene_geo_path.joinpath('%s_%s/geo.ply' % (args.scene_id, args.room_id))
        # move to current room centroid
        if not scene_geo_centered.exists():
            scene_mesh = trimesh.load(scene_geo)
            scene_mesh.vertices = scene_mesh.vertices + room_centroid_offset
            scene_mesh.export(scene_geo_centered)
        vis_args.append('scene_geo')
    else:
        scene_geo_centered = ''

    # vis_gt
    viser = VIS_Compare(gt_nodes=gt_object_nodes, pred_nodes=pred_object_nodes, room_bbox=gt_room_bbox,
                        skeleton_joints=skeleton_joints, keep_interact_skeleton=True, skip_rates=5,
                        scene_geo=str(scene_geo_centered))
    if args.pred_or_gt == 'gt':
        save_path = save_screen_shot_dir.joinpath('gt.jpg')
        viser.visualize(type=vis_args, save_path=str(save_path), offline=False, mode='gt')
    elif args.pred_or_gt == 'pred':
        save_path = save_screen_shot_dir.joinpath('pred.jpg')
        viser.visualize(type=vis_args, save_path=str(save_path), offline=False, mode='pred')
    else:
        raise NotImplementedError('Please specify if you want to visualize (pred) or (gt).')

