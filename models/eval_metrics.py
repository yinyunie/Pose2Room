#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
import torch

def calculate_scene_iou(pred_data, gt_data, normalization='sigmoid', iou_thresh=0.5):
    iou_scores = []
    pred_grids = []
    gt_grids = []
    for pred_scene_volume, gt_scene_volume in zip(pred_data, gt_data):
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            pred_scene_volume = torch.sigmoid(pred_scene_volume)
        elif normalization == 'softmax':
            pred_scene_volume = torch.softmax(pred_scene_volume, dim=1)
        else:
            pass
        pred_occ = pred_scene_volume > iou_thresh
        pred_occ = pred_occ.long().byte()  # convert to uint8
        gt_occ = gt_scene_volume.byte()
        iou_score = torch.sum(pred_occ & gt_occ).float() / torch.clamp(torch.sum(pred_occ | gt_occ).float(), min=1e-8)

        pred_grids.append(pred_occ.cpu().numpy())
        gt_grids.append(gt_occ.cpu().numpy())
        iou_scores.append(iou_score.item())
    return pred_grids, gt_grids, iou_scores

def calculate_iou(pred_data, gt_data, normalization='sigmoid', iou_thresh=0.5):
    n_batch, n_frame, n_channel, vx_size, vysize, vz_size = pred_data.shape
    pred_voxels = pred_data.view(n_batch * n_frame, n_channel, vx_size, vysize, vz_size)
    gt_voxels = gt_data['voxels'].view(n_batch * n_frame, n_channel, vx_size, vysize, vz_size)

    assert normalization in ['sigmoid', 'softmax', 'none']
    if normalization == 'sigmoid':
        pred_voxels = torch.sigmoid(pred_voxels)
    elif normalization == 'softmax':
        pred_voxels = torch.softmax(pred_voxels, dim=1)
    else:
        pass

    pred_occ = pred_voxels > iou_thresh
    pred_occ = pred_occ.long().byte() # convert to uint8
    gt_occ = gt_voxels.byte()

    pred_occ = pred_occ.view(n_batch * n_frame, -1)
    gt_occ = gt_occ.view(n_batch * n_frame, -1)

    iou_scores = torch.sum(pred_occ & gt_occ, dim=1).float() / torch.clamp(torch.sum(pred_occ | gt_occ, dim=1).float(), min=1e-8)

    return iou_scores
