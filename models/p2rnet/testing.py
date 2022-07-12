#  tester for P2RNet.
#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import torch
from models.testing import BaseTester
from .training import Trainer
import numpy as np
from net_utils.box_util import corners2params
from utils.pc_utils import rot2head
import os
from utils import pc_utils
from net_utils import utils
from utils.tools import write_json

class Tester(BaseTester, Trainer):
    '''
    Tester object for ISCNet.
    '''

    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        pass

    def evaluate_step(self, est_data, data):
        eval_metrics = {}
        return eval_metrics

    def test_step(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net.module.generate(data)

        '''computer losses'''
        loss = self.net.module.loss(est_data, data)
        eval_metrics = self.evaluate_step(est_data, data)

        # for logging
        loss_reduced = utils.reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        loss_dict = {**loss_dict, **eval_metrics}
        return loss_dict, est_data

    def visualize_step(self, phase, iter, gt_data, our_data):
        ''' Performs a visualization step.
        '''
        end_points, eval_dict, parsed_predictions = our_data
        batch_id = 0
        sample_name = gt_data['sample_idx'][batch_id]
        origin_joint_id = self.cfg.dataset_config.origin_joint_id

        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s'%(phase, iter, sample_name))
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        DUMP_CONF_THRESH = self.cfg.config['generation']['dump_threshold'] # Dump boxes with obj prob larger than that.

        '''Predict boxes'''
        pred_corners_3d = parsed_predictions['pred_corners_3d'][batch_id]
        objectness_prob = parsed_predictions['obj_prob'][batch_id]

        # INPUT
        input_joints = gt_data['input_joints'].cpu().numpy()

        # NETWORK OUTPUTS
        seed_xyz = end_points['seed_skeleton'][:, :, origin_joint_id].detach().cpu().numpy()  # (B,num_seed,3)
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()

        box_size, R_mat, center = corners2params(pred_corners_3d)
        heading = rot2head(R_mat)
        box_params = np.hstack([center, box_size, heading[:, np.newaxis]])

        # OTHERS
        pred_mask = eval_dict['pred_mask']  # B,num_proposal
        keep_idx = np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1)

        # # Dump input joints
        # _, idx = np.unique(input_joints[batch_id], axis=0, return_index=True)
        # input_joint_pnts = input_joints[batch_id][np.sort(idx)]
        # # pc_utils.write_joints(input_joint_pnts, output_file=os.path.join(dump_dir, '%06d_01_input_joints.obj' % (batch_id)))
        # all_joint_points = input_joint_pnts.reshape(input_joint_pnts.shape[0] * input_joint_pnts.shape[1], 3)
        # pc_utils.write_ply(all_joint_points, os.path.join(dump_dir, '%06d_01_input_joints.ply' % (batch_id)))
        #
        # # Dump seed_xyz
        # pc_utils.write_points(seed_xyz[batch_id], radius=0.05, color=(0.8, 0.6, 0.6),
        #                       output_file=os.path.join(dump_dir, '%06d_02_seeds.obj' % (batch_id)))

        # # Dump votes
        # if 'vote_xyz' in end_points:
        #     vote_xyz = end_points['vote_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
        #     pc_utils.write_points(vote_xyz[batch_id], radius=0.05, color=(0.8, 0.2, 0.2),
        #                           output_file=os.path.join(dump_dir, '%06d_03_reg_votes.obj' % (batch_id)))

        # # Dump aggregated_votes
        # pc_utils.write_points(aggregated_vote_xyz[batch_id], radius=0.05, color=(0.8, 0.1, 0.1),
        #                       output_file=os.path.join(dump_dir, '%06d_04_aggregated_votes.obj' % (batch_id)))

        # # Dump box centers
        # pc_utils.write_points(box_params[:, 0:3], radius=0.05, color=(0.2, 0.6, 0.2),
        #                       output_file=os.path.join(dump_dir, '%06d_05_proposal_centers.obj' % (batch_id)))

        # # Dump confident box centers
        # if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
        #     pc_utils.write_points(box_params[objectness_prob > DUMP_CONF_THRESH, 0:3], radius=0.05,
        #                           color=(0.1, 0.8, 0.1),
        #                           output_file=os.path.join(dump_dir,'%06d_06_confident_proposal_centers.obj' % (batch_id)))

        pred_sem_cls = parsed_predictions['pred_sem_cls'][batch_id]

        # # Dump all box proposals
        # if box_params.shape[0] > 0:
        #     pc_utils.write_oriented_bbox(box_params, pred_sem_cls, all_class_labels=self.cfg.dataset_config.class_labels,
        #                                  output_file=os.path.join(dump_dir, '%06d_07_pred_all_proposal_bbox.obj' % (batch_id)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            if box_params.shape[0] > 0:
                # pc_utils.write_oriented_bbox(
                #     box_params[keep_idx, :],
                #     pred_sem_cls[keep_idx],
                #     all_class_labels=self.cfg.dataset_config.class_labels,
                #     output_file=os.path.join(dump_dir, '%06d_08_pred_confident_nms_bbox.obj' % (batch_id)))
                save_path = os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.npz' % (batch_id))
                np.savez(save_path, obbs=box_params[keep_idx, :],
                         cls=pred_sem_cls[keep_idx], inst_idx=keep_idx)

        keep_idx = np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1)
        pi_file = os.path.join(dump_dir, '%06d_pi_dict.json' % (batch_id))
        pi_dict = {'center':end_points['pi']['center'][batch_id, : ,keep_idx].cpu().numpy().tolist(),
                   'size': end_points['pi']['size'][batch_id, :, keep_idx].cpu().numpy().tolist(),
                   'heading': end_points['pi']['heading'][batch_id, :, keep_idx].cpu().numpy().tolist()}
        write_json(pi_file, pi_dict)

        # Store GT outputs
        gt_center = gt_data['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_mask = gt_data['box_label_mask'].cpu().numpy()  # B,K2
        gt_size = torch.exp(gt_data['size']).detach().cpu().numpy()
        gt_sin_cos = gt_data['heading']
        gt_heading = torch.atan2(gt_sin_cos[..., 0], gt_sin_cos[..., 1]).detach().cpu().numpy()
        gt_sem_cls_label = gt_data['sem_cls_label'].cpu().numpy()

        # # Dump GT votes
        # seed_inds = end_points['seed_inds'].long()
        # seed_gt_votes_mask = torch.gather(gt_data['vote_label_mask'][..., origin_joint_id], 1, seed_inds)
        # GT_VOTE_FACTOR = 3# B,num_seed in [0,num_points-1]
        # batch_size, num_seed = seed_inds.size()
        # seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
        # seed_gt_votes = torch.gather(gt_data['vote_label'][:, :, origin_joint_id], 1, seed_inds_expand)
        # seed_gt_votes += end_points['seed_skeleton'][:,:,origin_joint_id].repeat(1, 1, 3)
        # seed_gt_votes = seed_gt_votes[batch_id, seed_gt_votes_mask[batch_id]>0]
        # seed_gt_votes_reshape = seed_gt_votes.view(1 * seed_gt_votes_mask.sum() * GT_VOTE_FACTOR, 3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
        # seed_gt_votes_reshape = seed_gt_votes_reshape.cpu().numpy()
        # pc_utils.write_points(seed_gt_votes_reshape, radius=0.05, color=(0.8, 0.2, 0.2),
        #                       output_file=os.path.join(dump_dir, '%06d_03_gt_votes.obj' % (batch_id)))

        # Dump GT bounding boxes
        obbs = []
        obb_classes = []
        for j in range(gt_center.shape[1]):
            if gt_mask[batch_id, j] == 0: continue
            obb = np.zeros((7,))
            obb[:3] = gt_center[batch_id, j, 0:3]
            obb[3:6] = gt_size[batch_id, j, 0:3]
            obb[6] = gt_heading[batch_id, j]
            obbs.append(obb)
            obb_classes.append(gt_sem_cls_label[batch_id, j])

        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            # pc_utils.write_oriented_bbox(obbs, obb_classes, all_class_labels=self.cfg.dataset_config.class_labels,
            #                              output_file=os.path.join(dump_dir, '%06d_09_gt_bbox.obj' % (batch_id)))
            save_path = os.path.join(dump_dir, '%06d_gt_bbox.npz' % (batch_id))
            np.savez(save_path, obbs=obbs, cls=obb_classes)
