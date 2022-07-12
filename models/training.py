#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
import torch
from net_utils import utils


class BaseTrainer(object):
    '''
    Base trainer for all networks.
    '''
    def __init__(self, cfg, net, optimizer, device=None):
        self.cfg = cfg
        self.net = net
        self.optimizer = optimizer
        self.device = device

    def show_lr(self):
        '''
        display current learning rates
        :return:
        '''
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        self.cfg.log_string('Current learning rates are: ' + str(lrs) + '.')

    def train_step(self, data):
        '''
        performs a step training
        :param data (dict): data dictionary
        :return:
        '''
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        if loss['total'].requires_grad:
            loss['total'].backward()
            if self.cfg.config['optimizer']['clip_norm'] > 0:
                max_norm = self.cfg.config['optimizer']['clip_norm']
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm)
            self.optimizer.step()

        # for logging
        loss_reduced = utils.reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        return loss_dict

    def eval_loss_parser(self, loss_recorder):
        '''
        get the eval
        :param loss_recorder: loss recorder for all losses.
        :return:
        '''
        return loss_recorder['total'].avg

    def compute_loss(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize_step(self, *args, **kwargs):
        ''' Performs a visualization step.
        '''
        if not self.cfg.config['device']['is_main_process']:
            return
        raise NotImplementedError
