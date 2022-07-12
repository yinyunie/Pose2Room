#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
class BaseTester(object):
    '''
    Base tester for all networks.
    '''
    def __init__(self, cfg, net, device=None):
        self.cfg = cfg
        self.net = net
        self.device = device

    def visualize_step(self, *args, **kwargs):
        ''' Performs a visualization step.
        '''
        if not self.cfg.config['device']['is_main_process']:
            return
        raise NotImplementedError

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        # camera orientation error
        raise NotImplementedError




