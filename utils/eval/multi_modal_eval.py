from pathlib import Path
from dateutil.parser import isoparse
import numpy as np
from utils.pc_utils import head2rot
from utils.tools import get_box_corners
import re
from scipy.stats import entropy


def read_pred(sample_filename):
    bbox_info = np.load(sample_filename)
    inst_inds = np.where(bbox_info['inst_idx'])[0]
    object_nodes = []
    for inst_id, bbox, cls_label in zip(inst_inds, bbox_info['obbs'], bbox_info['cls']):
        centroid = bbox[:3]
        box_size = bbox[3:6]
        heading_angle = bbox[6]
        R_mat = head2rot(heading_angle)

        object_node = {}
        object_node['centroid'] = centroid
        object_node['R_mat'] = R_mat
        object_node['size'] = box_size
        object_node['class_id'] = cls_label
        object_node['inst_id'] = inst_id
        object_nodes.append(object_node)
    return object_nodes


def det_sigma(data_term):
    data_term = np.array(data_term)
    if len(data_term.shape) == 1:
        return np.var(data_term)
    return np.linalg.det(np.cov(data_term, rowvar=False))

if __name__ == '__main__':
    root_dir = Path('out/p2rnet/test')
    eval_path_start_end = ['2021-10-28T13:05:13.348486', '2021-10-28T13:34:08.044815']
    all_dirs = list(root_dir.iterdir())
    eval_dirs = []
    for dirname in all_dirs:
        try:
            if isoparse(dirname.name) >= isoparse(eval_path_start_end[0]) and isoparse(dirname.name) <= isoparse(
                eval_path_start_end[1]):
                eval_dirs.append(dirname)
        except ValueError:
            continue
    eval_dirs = np.random.choice(eval_dirs, 10, replace=False)
    eval_dirs.sort()

    # map max
    map_score_list = []
    for eval_dir in eval_dirs:
        log_file = eval_dir.joinpath('log.txt')
        log_handle = open(log_file, 'r')
        log_info = log_handle.read()
        log_handle.close()
        map_scores = re.findall(r"eval mAP: (\d+\.\d+)\n", log_info)
        map_scores = [float(score) for score in map_scores]
        map_score_list.append(map_scores)
    print(np.max(map_score_list, axis=0))

    # diversity
    sample_dirnames = eval_dirs[0].joinpath('visualization').iterdir()
    sample_dirnames = [subdir.name for subdir in sample_dirnames]

    stat_dict = dict()
    for sample_dirname in sample_dirnames:
        if sample_dirname not in stat_dict:
            stat_dict[sample_dirname] = {}
        for eval_dir in eval_dirs:
            data_path = eval_dir.joinpath('visualization', sample_dirname, '000000_pred_confident_nms_bbox.npz')
            assert data_path.exists()
            save_data = read_pred(data_path)
            for instance in save_data:
                inst_id = instance['inst_id']
                # recover bbox
                centroid = instance['centroid']
                vectors = np.diag(np.array(instance['size']) / 2.).dot(instance['R_mat'])
                box_corners = np.array(get_box_corners(centroid, vectors))

                if inst_id not in stat_dict[sample_dirname]:
                    stat_dict[sample_dirname][inst_id] = {'box3d': [],
                                                          'class_id': []}
                stat_dict[sample_dirname][inst_id]['box3d'].append(box_corners)
                stat_dict[sample_dirname][inst_id]['class_id'].append(instance['class_id'])

    TMD = []
    for sample_dirname, inst_stat in stat_dict.items():
        for inst_id, per_inst_stat in inst_stat.items():
            _, freq = np.unique(per_inst_stat['class_id'],return_counts=True)
            # Shannon's entropy
            cls_entropy = entropy(freq/sum(freq), base=2)
            box3ds_per_inst = np.array(per_inst_stat['box3d'])
            pair_wise_dist = box3ds_per_inst[:, np.newaxis] - box3ds_per_inst[np.newaxis]
            pair_wise_dist = np.mean(np.linalg.norm(pair_wise_dist, axis=-1), axis=-1)
            shape_variance = np.mean(pair_wise_dist.sum(axis=-1))
            dist = (cls_entropy + 1) * (shape_variance + 1)
            TMD.append(dist)

    print(np.mean(TMD))
