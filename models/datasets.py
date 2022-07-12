#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import os
from torch.utils.data import Dataset
from utils.tools import read_json


class Base_Dataset(Dataset):
    def __init__(self, cfg, mode):
        '''
        initiate a base dataset for data loading in other networks
        :param cfg: config file
        :param mode: train/val/test mode
        '''
        self.config = cfg.config
        self.dataset_config = cfg.dataset_config
        self.mode = mode
        split_file = os.path.join(cfg.config['data']['split'], mode + '.json')
        self.split = read_json(split_file)


    def __len__(self):
        return len(self.split)
