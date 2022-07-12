#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import os
import yaml
import logging
from datetime import datetime


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def read_to_dict(input):
    if not input:
        return dict()
    if isinstance(input, str) and os.path.isfile(input):
        if input.endswith('yaml'):
            with open(input, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            ValueError('Config file should be with the format of *.yaml')
    elif isinstance(input, dict):
        config = input
    else:
        raise ValueError('Unrecognized input type (i.e. not *.yaml file nor dict).')

    return config

class CONFIG(object):
    '''
    Stores all configures
    '''
    def __init__(self, args, config):
        '''
        Loads config file
        @param args: input args
        @param config (dict): config file
        @return:
        '''
        self.config = config
        self.is_main_process = config['device']['is_main_process']
        self._logger, self._save_path = self.load_logger(args.mode)

        # update save_path to config file
        self.update_config(log={'path': self._save_path})

        # update visualization path
        if self.is_main_process:
            vis_path = os.path.join(self._save_path, self.config['log']['vis_path'])
            if not os.path.exists(vis_path):
                os.mkdir(vis_path)
        else:
            vis_path = ''

        self.update_config(log={'vis_path': vis_path})

    @property
    def logger(self):
        return self._logger

    @property
    def save_path(self):
        return self._save_path

    def load_logger(self, mode='train'):
        if not self.is_main_process:
            return None, ''
        # set file handler
        save_path = os.path.join(self.config['log']['path'], mode, datetime.now().isoformat())
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logfile = os.path.join(save_path, 'log.txt')
        file_handler = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.__file_handler = file_handler

        # configure logger
        logger = logging.getLogger('Empty')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        return logger, save_path

    def log_string(self, content):
        if self.is_main_process:
            self._logger.info(content)

    def update_config(self, *args, **kwargs):
        '''
        update config and corresponding logger setting
        :param input: dict settings add to config file
        :return:
        '''
        cfg1 = dict()
        for item in args:
            cfg1.update(read_to_dict(item))

        cfg2 = read_to_dict(kwargs)

        new_cfg = {**cfg1, **cfg2}

        update_recursive(self.config, new_cfg)
        # when update config file, the corresponding logger should also be updated.
        self.__update_logger()

    def write_config(self):
        if not self.is_main_process:
            return

        output_file = os.path.join(self._save_path, 'out_config.yaml')
        with open(output_file, 'w') as file:
            yaml.dump(self.config, file, default_flow_style = False)

    def __update_logger(self):
        if not self.is_main_process:
            return

        # configure logger
        name = self.config['mode'] if 'mode' in self.config else self._logger.name
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.__file_handler)
        self._logger = logger

def mount_external_config(cfg):
    # mount external data for network forward pass.
    from configs.dataset_config import Dataset_Config
    dataset_cfg = Dataset_Config(cfg.config['data']['dataset'])
    setattr(cfg, 'dataset_config', dataset_cfg)

    # Used for AP calculation
    if cfg.config['mode'] != 'train':
        eval_cfg = cfg.config['test']
        CONFIG_DICT = {'remove_far_box': eval_cfg['remove_far_box'],
                       'use_3d_nms': eval_cfg['use_3d_nms'],
                       'nms_iou': eval_cfg['nms_iou'],
                       'use_old_type_nms': eval_cfg['use_old_type_nms'],
                       'cls_nms': eval_cfg['use_cls_nms'],
                       'per_class_proposal': eval_cfg['per_class_proposal'],
                       'conf_thresh': eval_cfg['conf_thresh'],
                       'multi_mode': eval_cfg['multi_mode'],
                       'sample_cls': eval_cfg['sample_cls'],
                       'dataset_config': dataset_cfg}
        setattr(cfg, 'eval_config', CONFIG_DICT)
    return cfg
