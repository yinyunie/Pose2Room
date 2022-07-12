#  Copyright (c) 11.2021. Yinyu Nie
#  License: MIT

from net_utils.utils import load_device, load_model
from net_utils.utils import CheckpointIO
from configs.config_utils import mount_external_config
import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
from models.p2rnet.dataloader import my_worker_init_fn, collate_fn
from pathlib import Path
from net_utils.box_util import corners2params
from utils import pc_utils
from utils.pc_utils import rot2head, head2rot
from utils.vis_base import VIS_BASE
from utils.virtualhome import dataset_config, LIMBS, valid_joint_ids
from utils.virtualhome.vis_gt_vh import dist_node2bbox, get_even_dist_joints
import seaborn as sns
import vtk

class Demo_DataSet(Dataset):
    def __init__(self, cfg):
        self.num_frames = cfg.config['data']['num_frames']
        self.use_height = not cfg.config['data']['no_height']
        self.split = list(Path(cfg.config['demo_path']).joinpath('inputs').iterdir())

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        '''Get each sample'''
        '''Load data'''
        data_path = self.split[idx]
        skeleton_joints = np.load(str(data_path))

        if self.use_height:
            floor_height = np.percentile(skeleton_joints[..., 1], 0.99)
            height = skeleton_joints[..., 1] - floor_height
            skeleton_joints = np.concatenate([skeleton_joints, np.expand_dims(height, -1)], -1)

        # Process input frames
        joint_ids = np.linspace(0, skeleton_joints.shape[0]-1, self.num_frames).round().astype(np.uint16)
        input_joints = skeleton_joints[joint_ids]

        # deliver to network
        ret_dict = {}
        ret_dict['input_joints'] = input_joints.astype(np.float32)
        ret_dict['sample_idx'] = '.'.join(data_path.name.split('.')[:-1])
        return ret_dict

def load_dataloader(cfg, mode='test'):
    dataset = Demo_DataSet(cfg)
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=cfg.config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)
    return dataloader

class Vis_Demo(VIS_BASE):
    def __init__(self, pred_nodes=(), skeleton_joints=None, skip_rates=1, keep_interact_skeleton=False):
        super(Vis_Demo, self).__init__()
        self.pred_nodes = pred_nodes
        self.pred_class_ids, self.pred_palette_cls = self.get_cls_palatte(pred_nodes)

        self.move_traj = skeleton_joints[:, 0]
        selected_sk_ids = range(skeleton_joints.shape[0])
        if skip_rates > 1 and not keep_interact_skeleton:
            selected_sk_ids = get_even_dist_joints(skeleton_joints, skip_rates)
            skeleton_joints = skeleton_joints[selected_sk_ids]
        elif keep_interact_skeleton:
            joint_coordinates = skeleton_joints.reshape(-1, 3)
            # get distance between joint to nodes
            selected_sk_ids = dist_node2bbox(pred_nodes, joint_coordinates, dataset_config.joint_num)
            # add more frames close to it.
            if skip_rates == 1:
                local_sk_ids = np.arange(-20, 20, skip_rates)[np.newaxis]
                selected_sk_ids = selected_sk_ids[:, np.newaxis] + local_sk_ids
                selected_sk_ids = selected_sk_ids.flatten()
                selected_sk_ids = selected_sk_ids[selected_sk_ids < skeleton_joints.shape[0]]
                selected_sk_ids = np.sort(selected_sk_ids)
            else:
                local_sk_ids = np.arange(-20, 20)[np.newaxis]
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

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        cam_fp = (self.move_traj.max(0) + self.move_traj.min(0)) / 2.
        cam_loc = cam_fp + kwargs.get('cam_centroid', [5, 3, 5])
        cam_up = [0, sum((cam_loc - cam_fp) ** 2) / (cam_loc[1] - cam_fp[1]), 0] + cam_fp - cam_loc
        camera = self.set_camera(cam_loc, cam_fp, cam_up, self.cam_K)
        renderer.SetActiveCamera(camera)

        '''draw 3D boxes'''
        if 'bboxes' in kwargs['type']:
            vis_nodes = self.pred_nodes
            class_ids = self.pred_class_ids
            palette_cls = self.pred_palette_cls
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

    def get_cls_palatte(self, nodes):
        if len(nodes):
            class_ids = [node['class_id'][0] for node in nodes]
            # set palette
            palette_cls = np.array([*sns.color_palette("hls", len(dataset_config.class_labels))])
            return class_ids, palette_cls
        else:
            return None, None

def visualize_step(cfg, phase, iter, gt_data, our_data):
    ''' Performs a visualization step.
    '''
    end_points, eval_dict, parsed_predictions = our_data
    batch_id = 0
    sample_name = gt_data['sample_idx'][batch_id]
    dump_dir = Path(cfg.config['demo_path']).joinpath('outputs').joinpath(sample_name)
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True)

    DUMP_CONF_THRESH = cfg.config['generation']['dump_threshold'] # Dump boxes with obj prob larger than that.

    '''Predict boxes'''
    pred_corners_3d = parsed_predictions['pred_corners_3d'][batch_id]
    objectness_prob = parsed_predictions['obj_prob'][batch_id]

    # INPUT
    input_joints = gt_data['input_joints'].cpu().numpy()

    # NETWORK OUTPUTS
    box_size, R_mat, center = corners2params(pred_corners_3d)
    heading = rot2head(R_mat)
    box_params = np.hstack([center, box_size, heading[:, np.newaxis]])

    # OTHERS
    pred_mask = eval_dict['pred_mask']  # B,num_proposal
    keep_idx = np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1)


    '''Visualize results'''
    _, idx = np.unique(input_joints[batch_id], axis=0, return_index=True)
    input_joint_pnts = input_joints[batch_id][np.sort(idx)]

    pred_sem_cls = parsed_predictions['pred_sem_cls'][batch_id]

    inst_bboxes = box_params[keep_idx, :]
    inst_labels = pred_sem_cls[keep_idx]

    object_nodes = []
    for bbox, cls_label in zip(inst_bboxes, inst_labels):
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

    viser = Vis_Demo(skeleton_joints=input_joint_pnts, pred_nodes=object_nodes, skip_rates=10, keep_interact_skeleton=True)
    viser.visualize(type=['bboxes', 'skeleton'])


def predict(cfg, demo_loader, net, device):
    data = next(iter(demo_loader))

    data['input_joints'] = data['input_joints'].to(device)
    est_data = net.module.generate(data, eval=False)
    # visualize intermediate results.
    visualize_step(cfg, 'demo', 0, data, est_data)


def run(cfg):
    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    cfg = mount_external_config(cfg)

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load existing checkpoint'''
    checkpoint.parse_checkpoint()

    '''Load data'''
    cfg.log_string('Loading dataset.')
    demo_loader = load_dataloader(cfg, mode='test')

    '''Start to predict'''
    cfg.log_string('Start to test.')
    cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))

    net.train(cfg.config['mode'] == 'train')
    with torch.no_grad():
        predict(cfg=cfg, demo_loader=demo_loader, net=net, device=device)

    cfg.write_config()
    cfg.log_string('Testing finished.')
