#  Split samples for training, testing and validation
#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT
import random
import numpy as np
from utils.virtualhome import dataset_config
from utils.tools import write_json, read_json
from utils.virtualhome.vhome_utils import class_mapping, category_mapping


def get_bbox_layout_sizes(room_marks):
    room_marks = [mark.split('_') for mark in room_marks]
    # Get object bbox sizes
    obj_sizes_cls = {}
    for cls_id in range(len(dataset_config.class_labels)):
        obj_sizes_cls[cls_id] = []
    obj_sizes_category = {}
    for cat_id in range(len(dataset_config.category_labels)):
        obj_sizes_category[cat_id] = []

    obj_node_files = [dataset_config.script_bbox_path.joinpath(mark[0], 'bbox_%s.json' % (mark[1])) for mark in room_marks]
    for obj_node_file in obj_node_files:
        obj_nodes = read_json(obj_node_file)
        for node in obj_nodes:
            class_id = class_mapping([node['class_name']])[0][0]
            category_id = category_mapping([node['class_name']])[0][0]
            obj_sizes_cls[class_id].append(node['size'])
            obj_sizes_category[category_id].append(node['size'])

    # Get layout sizes
    layout_sizes = []
    layout_files = [dataset_config.script_bbox_path.joinpath(mark[0], 'room_bbox_%s.json' % (mark[1])) for mark in
                    room_marks]
    for layout_file in layout_files:
        layout_sizes.append(read_json(layout_file)['room_bbox']['size'])
    return obj_sizes_cls, obj_sizes_category, layout_sizes

if __name__ == '__main__':
    np.random.seed(dataset_config.random_seed)
    random.seed(dataset_config.random_seed)

    all_sample_files = list(dataset_config.sample_path.rglob('*.hdf5'))

    split_level = dataset_config.split_level
    split_raio = dataset_config.split_ratio[split_level]

    if split_level == 'room_level':
        # Asking to generalize to different rooms
        split_dir = dataset_config.split_path.joinpath(split_level)
        if not split_dir.is_dir():
            split_dir.mkdir()
        sample_marks = ['_'.join(file.name.split('_')[:2]) for file in all_sample_files]
        unique_marks, reverse_idx = np.unique(sample_marks, return_inverse=True)
        mark_num = len(unique_marks)
        train_num = np.round(mark_num * dataset_config.split_ratio[split_level]['train']).astype(np.uint32)
        val_num = np.round(mark_num * dataset_config.split_ratio[split_level]['val']).astype(np.uint32)
        sample_ordering = np.random.permutation(mark_num)
        train_ids = sample_ordering[:train_num]
        val_ids = sample_ordering[train_num: train_num + val_num]

        train_files = []
        train_marks = []
        val_files = []

        for sample_idx, mark_id in enumerate(reverse_idx):
            if 'Female2' not in str(all_sample_files[sample_idx]):
                continue
            if mark_id in train_ids:
                train_files.append(str(all_sample_files[sample_idx]))
                train_marks.append(['_'.join(all_sample_files[sample_idx].name.split('_')[:2])] + all_sample_files[sample_idx].name.split('_')[2:4])
            elif mark_id in val_ids:
                val_files.append(str(all_sample_files[sample_idx]))

        split_data = {'train': train_files,
                      'test': val_files,
                      'val': val_files}
        for key, value in split_data.items():
            write_json(split_dir.joinpath(key + '.json'), value)
    elif split_level == 'char_level':
        # Asking to generalize to different characters
        split_dir = dataset_config.split_path.joinpath(split_level)
        if not split_dir.is_dir():
            split_dir.mkdir()
        sample_marks = [file.name.split('_')[3] for file in all_sample_files]
        unique_marks, reverse_idx = np.unique(sample_marks, return_inverse=True)
        mark_num = len(unique_marks)
        train_num = np.round(mark_num * dataset_config.split_ratio[split_level]['train']).astype(np.uint32)
        val_num = np.round(mark_num * dataset_config.split_ratio[split_level]['val']).astype(np.uint32)
        sample_ordering = np.random.permutation(mark_num)
        train_ids = sample_ordering[:train_num]
        val_ids = sample_ordering[train_num: train_num + val_num]

        train_files = []
        train_marks = []
        val_files = []
        for sample_idx, mark_id in enumerate(reverse_idx):
            if mark_id in train_ids:
                train_files.append(str(all_sample_files[sample_idx]))
                train_marks.append(['_'.join(all_sample_files[sample_idx].name.split('_')[:2])] + all_sample_files[sample_idx].name.split('_')[2:4])
            elif mark_id in val_ids:
                val_files.append(str(all_sample_files[sample_idx]))

        split_data = {'train': train_files,
                      'test': val_files,
                      'val': val_files}
        for key, value in split_data.items():
            write_json(split_dir.joinpath(key + '.json'), value)

    elif split_level == 'room_char_level':
        # Asking to generalize to different rooms and different characters
        raise NotImplementedError
    elif split_level == 'script_level':
        # Split data purely based on scripts
        split_dir = dataset_config.split_path.joinpath(split_level)
        if not split_dir.is_dir():
            split_dir.mkdir()
        sample_marks = [['_'.join(file.name.split('_')[:2])] + [file.name.split('_')[2]] for file in
                        all_sample_files]
        unique_marks, reverse_idx = np.unique(sample_marks, axis=0, return_inverse=True)
        mark_num = unique_marks.shape[0]
        train_num = np.round(dataset_config.split_ratio[split_level]['train'] * mark_num).astype(np.uint32)
        val_num = np.round(mark_num * dataset_config.split_ratio[split_level]['val']).astype(np.uint32)
        sample_ordering = np.random.permutation(mark_num)
        train_ids = sample_ordering[:train_num]
        val_ids = sample_ordering[train_num: train_num + val_num]

        train_marks = unique_marks[train_ids].tolist()
        train_files = []
        val_files = []

        for sample_idx, mark_id in enumerate(reverse_idx):
            if 'Female2' not in str(all_sample_files[sample_idx]):
                continue
            if mark_id in train_ids:
                train_files.append(str(all_sample_files[sample_idx]))
            else:
                val_files.append(str(all_sample_files[sample_idx]))

        split_data = {'train': train_files,
                      'test': val_files,
                      'val': val_files}
        for key, value in split_data.items():
            write_json(split_dir.joinpath(key + '.json'), value)

    else:
        raise NotImplementedError

    '''Get average bbox and layout sizes'''
    room_marks_in_training = set([item[0] for item in train_marks])
    obj_sizes_cls, obj_sizes_category, layout_sizes = get_bbox_layout_sizes(room_marks_in_training)

    '''save priors'''
    # for class-level labels
    obj_size_cls_avg = {}
    for cls_id, cls_sizes in obj_sizes_cls.items():
        if len(cls_sizes):
            obj_size_cls_avg[cls_id] = np.mean(cls_sizes, axis=0).tolist()

    # fill in those blank classes
    obj_size_cls_mean = np.mean([value for value in obj_size_cls_avg.values() if value], axis=0)
    for cls_id in obj_sizes_cls.keys():
        if cls_id not in obj_size_cls_avg:
            obj_size_cls_avg[cls_id] = obj_size_cls_mean.tolist()

    # for category-level labels
    obj_size_category_avg = {}
    for cat_id, cat_sizes in obj_sizes_category.items():
        if len(cat_sizes):
            obj_size_category_avg[cat_id] = np.mean(cat_sizes, axis=0).tolist()

    # fill in those blank categories
    obj_size_category_mean = np.mean([value for value in obj_size_category_avg.values() if value], axis=0)
    for cat_id in obj_sizes_category.keys():
        if cat_id not in obj_size_category_avg:
            obj_size_category_avg[cat_id] = obj_size_category_mean.tolist()

    prior_data = {'obj_size_cls_avg': obj_size_cls_avg,
                  'obj_size_category_avg': obj_size_category_avg,
                  'layout_size_avg': np.mean(layout_sizes, axis=0).tolist()}
    write_json(dataset_config.prior_path, prior_data)