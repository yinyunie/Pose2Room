#  P2RNet: model loader
#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork
import torch
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls

@METHODS.register_module
class P2RNet(BaseNetwork):
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        phase_names = []
        if cfg.config[cfg.config['mode']]['phase'] in ['full']:
            phase_names += ['backbone', 'centervoting', 'detection']

        if (not cfg.config['model']) or (not phase_names):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1), cfg.config['device']['gpu'], cfg))

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def generate(self, data, eval=True):
        '''
        Forward pass of the network for object detection
        '''
        # --------- Backbone ---------
        end_points = {}
        end_points = self.backbone(data['input_joints'], end_points)
        xyz = end_points['seed_skeleton']
        features = end_points['seed_features']

        # --------- Generate Center Candidates ---------
        xyz, features = self.centervoting(xyz, features)
        features_norm = torch.norm(features, p=2, dim=2)
        features = features.div(features_norm.unsqueeze(2))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        # --------- DETECTION ---------
        end_points, _ = self.detection.generate(xyz, features, end_points, False)
        eval_dict, parsed_predictions = parse_predictions(end_points, data, self.cfg.eval_config)
        eval_dict = assembly_pred_map_cls(eval_dict, parsed_predictions, self.cfg.eval_config)

        if eval:
            '''Get meta data for evaluation'''
            parsed_gts = parse_groundtruths(data, self.cfg.eval_config)

            '''for mAP evaluation'''
            eval_dict['batch_gt_map_cls'] = assembly_gt_map_cls(parsed_gts)

        return end_points, eval_dict, parsed_predictions

    def forward(self, data):
        '''
        Forward pass of the network
        :param data (dict): contains the data for training.
        :return: end_points: dict
        '''
        # --------- Backbone ---------
        end_points = {}
        end_points = self.backbone(data['input_joints'], end_points)
        xyz = end_points['seed_skeleton']
        features = end_points['seed_features']

        # --------- Generate Center Candidates ---------
        xyz, features = self.centervoting(xyz, features)
        features_norm = torch.norm(features, p=2, dim=2)
        features = features.div(features_norm.unsqueeze(2))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        # --------- DETECTION ---------
        end_points, _ = self.detection(xyz, features, end_points, False)
        return end_points

    def loss(self, pred_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        if isinstance(pred_data, tuple):
            pred_data = pred_data[0]

        total_loss = self.detection_loss(pred_data, gt_data, self.cfg.dataset_config)
        return total_loss
