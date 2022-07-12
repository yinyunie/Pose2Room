#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import os
import urllib
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import model_zoo
import torch.distributed as dist
from models.registers import METHODS
import sys
from models import method_paths
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import subprocess


class CheckpointIO(object):
    '''
    load, save, resume network weights.
    '''
    def __init__(self, cfg, **kwargs):
        '''
        initialize model and optimizer.
        :param cfg: configuration file
        :param kwargs: model, optimizer and other specs.
        '''
        self.cfg = cfg
        self._module_dict = kwargs
        self._module_dict.update({'epoch': 0, 'min_loss': 1e8})
        self._saved_filename = 'model_last.pth'

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def saved_filename(self):
        return self._saved_filename

    @staticmethod
    def is_url(url):
        scheme = urllib.parse.urlparse(url).scheme
        return scheme in ('http', 'https')

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self._module_dict.update(kwargs)

    def save(self, suffix=None, **kwargs):
        '''
        save the current module dictionary.
        :param kwargs:
        :return:
        '''
        if not self.cfg.config['device']['is_main_process']:
            return

        outdict = kwargs
        for k, v in self._module_dict.items():
            if hasattr(v, 'state_dict'):
                outdict[k] = v.state_dict()
            else:
                outdict[k] = v

        if not suffix:
            filename = self.saved_filename
        else:
            filename = self.saved_filename.replace('last', suffix)

        torch.save(outdict, os.path.join(self.cfg.config['log']['path'], filename))

    def load(self, filename, *domain):
        '''
        load a module dictionary from local file or url.
        :param filename (str): name of saved module dictionary
        :return:
        '''

        if self.is_url(filename):
            return self.load_url(filename, *domain)
        else:
            return self.load_file(filename, *domain)

    def parse_checkpoint(self):
        '''
        check if resume or finetune from existing checkpoint.
        :return:
        '''
        if self.cfg.config['resume']:
            # resume everything including net weights, optimizer, last epoch, last loss.
            self.cfg.log_string('Begin to resume from the last checkpoint.')
            self.resume()
        elif self.cfg.config['finetune']:
            # only load net weights.
            self.cfg.log_string('Begin to finetune from the existing weight.')
            self.finetune()
        else:
            self.cfg.log_string('Begin to train from scratch.')

    def finetune(self):
        '''
        finetune fron existing checkpoint
        :return:
        '''
        if isinstance(self.cfg.config['weight'], str):
            weight_paths = [self.cfg.config['weight']]
        else:
            weight_paths = self.cfg.config['weight']

        for weight_path in weight_paths:
            if not os.path.exists(weight_path):
                self.cfg.log_string('Warning: finetune failed: the weight path %s is invalid. Begin to train from scratch.' % (weight_path))
            else:
                self.load(weight_path, 'net')
                self.cfg.log_string('Weights for finetuning loaded.')

    def resume(self):
        '''
        resume the lastest checkpoint
        :return:
        '''
        checkpoint_root = os.path.dirname(self.cfg.save_path)
        saved_log_paths = os.listdir(checkpoint_root)
        saved_log_paths.sort(reverse=True)

        for last_path in saved_log_paths:
            last_checkpoint = os.path.join(checkpoint_root, last_path, self.saved_filename)
            if not os.path.exists(last_checkpoint):
                continue
            else:
                self.load(last_checkpoint)
                self.cfg.log_string('Last checkpoint resumed.')
                return

        self.cfg.log_string('Warning: resume failed: No checkpoint available. Begin to train from scratch.')

    def load_file(self, filename, *domain):
        '''
        load a module dictionary from file.
        :param filename: name of saved module dictionary
        :return:
        '''

        if os.path.exists(filename):
            self.cfg.log_string('Loading checkpoint from %s.' % (filename))
            checkpoint = torch.load(filename)
            scalars = self.parse_state_dict(checkpoint, *domain)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, *domain):
        '''
        load a module dictionary from url.
        :param url: url to a saved model
        :return:
        '''
        self.cfg.log_string('Loading checkpoint from %s.' % (url))
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict, domain)
        return scalars

    def parse_state_dict(self, checkpoint, *domain):
        '''
        parse state_dict of model and return scalars
        :param checkpoint: state_dict of model
        :return:
        '''
        for key, value in self._module_dict.items():

            # only load specific key names.
            if domain and (key not in domain):
                continue

            if key in checkpoint:
                if hasattr(value, 'load_state_dict'):
                    if key != 'net':
                        value.load_state_dict(checkpoint[key])
                    else:
                        '''load weights module by module'''
                        value.module.load_weight(checkpoint[key])
                else:
                    self._module_dict.update({key: checkpoint[key]})
            else:
                self.cfg.log_string('Warning: Could not find %s in checkpoint!' % key)

        if not domain:
            # remaining weights in state_dict that not found in our models.
            scalars = {k:v for k,v in checkpoint.items() if k not in self._module_dict}
            if scalars:
                self.cfg.log_string('Warning: the remaining modules %s in checkpoint are not found in our current setting.' % (scalars.keys()))
        else:
            scalars = {}

        return scalars

def initiate_environment(config):
    '''
    initiate randomness.
    :param config:
    :return:
    '''
    config = init_distributed_mode(config)
    seed = config['seed'] + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    return config

def load_device(cfg):
    '''
    load device settings
    :param config:
    :return:
    '''
    if cfg.config['device']['use_gpu'] and torch.cuda.is_available():
        cfg.log_string('GPU mode is on.')
        cfg.log_string('GPU Ids: %s used.' % (cfg.config['device']['gpu_ids']))
        return torch.device("cuda")
    else:
        cfg.log_string('CPU mode is on.')
        return torch.device("cpu")

def load_model(cfg, device):
    '''
    load specific network from configuration file
    :param config: configuration file
    :param device: torch.device
    :return:
    '''
    if cfg.config['method'] not in METHODS.module_dict:
        cfg.log_string('The method %s is not defined, please check the correct name.' % (cfg.config['method']))
        cfg.log_string('Exit now.')
        sys.exit(0)

    model = METHODS.get(cfg.config['method'])(cfg)
    model.to(device)

    if cfg.config['device']['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.config['device']['gpu']])
    else:
        model = nn.DataParallel(model, device_ids=[cfg.config['device']['gpu']])

    return model

def load_trainer(cfg, net, optimizer, device):
    '''
    load trainer for training and validation
    :param cfg: configuration file
    :param net: nn.Module network
    :param optimizer: torch.optim
    :param device: torch.device
    :return:
    '''
    trainer = method_paths[cfg.config['method']].config.get_trainer(cfg=cfg,
                                                                    net=net,
                                                                    optimizer=optimizer,
                                                                    device=device)
    return trainer

def load_tester(cfg, net, device):
    '''
    load tester for testing
    :param cfg: configuration file
    :param net: nn.Module network
    :param device: torch.device
    :return:
    '''
    tester = method_paths[cfg.config['method']].config.get_tester(cfg=cfg,
                                                                  net=net,
                                                                  device=device)
    return tester

def load_dataloader(cfg, mode):
    '''
    load dataloader
    :param cfg: configuration file.
    :param mode: 'train', 'val' or 'test'.
    :return:
    '''
    dataloader = method_paths[cfg.config['method']].config.get_dataloader(cfg=cfg,
                                                                          mode=mode)
    return dataloader

class AverageMeter(object):
    '''
    Computes ans stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val # current value
        if not isinstance(val, list):
            self.sum += val * n # accumulated sum, n = batch_size
            self.count += n # accumulated count
        else:
            self.sum += sum(val)
            self.count += len(val)
        self.avg = self.sum / self.count # current average value

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]

class LossRecorder(object):
    def __init__(self, batch_size=1):
        '''
        Log loss data
        :param config: configuration file.
        :param phase: train, validation or test.
        '''
        self._batch_size = batch_size
        self._loss_recorder = {}

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def loss_recorder(self):
        return self._loss_recorder

    def update_loss(self, loss_dict):
        for key, item in loss_dict.items():
            if key not in self._loss_recorder:
                self._loss_recorder[key] = AverageMeter()
            self._loss_recorder[key].update(item, self._batch_size)

    def synchronize_between_processes(self):
        for meter in self._loss_recorder.values():
            meter.synchronize_between_processes()

class LogBoard(object):
    def __init__(self, cfg):
        self.is_main_process = cfg.config['device']['is_main_process']
        if not self.is_main_process:
            return
        self.writer = SummaryWriter()
        self.iter = {'train': 1, 'val': 1, 'test': 1 }

    def update(self, value_dict, step_len, phase):
        if not self.is_main_process:
            return
        n_iter = self.iter[phase] * step_len
        for key, item in value_dict.items():
            self.writer.add_scalar(key + '/' + phase, item, n_iter)
        self.iter[phase] += 1

    def plot_gradients(self, net):
        if not self.is_main_process:
            return
        ave_grads = []
        max_grads = []
        layers = []
        for name, param in net.named_parameters():
            if (param.requires_grad) and ("bias" not in name):
                layers.append('.'.join(name.split('.')[1:]))
                ave_grads.append(param.grad.abs().mean())
                max_grads.append(param.grad.abs().max())
        layers = layers[::-1]
        ave_grads = ave_grads[::-1]
        max_grads = max_grads[::-1]
        fig = plt.figure(figsize=(12, 16))
        ax = fig.add_subplot(111)
        plt.barh(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
        plt.barh(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
        plt.vlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.yticks(range(0, len(ave_grads), 1), layers)
        plt.ylim(bottom=0, top=len(ave_grads))
        plt.xlim(left=-0.001, right=0.02)  # zoom in on the lower gradient regions
        plt.ylabel("Layers")
        plt.xlabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        xleft, xright = ax.get_xlim()
        fig.tight_layout()
        ax.set_aspect(abs((xright-xleft)/len(ave_grads)) * 2)
        self.writer.add_figure('matplotlib', fig)


# For distributed data parallel
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config['device']['rank'] = int(os.environ["RANK"])
        config['device']['world_size'] = int(os.environ['WORLD_SIZE'])
        config['device']['gpu'] = int(os.environ['LOCAL_RANK'])
    # elif 'SLURM_PROCID' in os.environ:
    #     config['device']['rank'] = int(os.environ['SLURM_PROCID'])
    #     config['device']['gpu'] = config['device']['rank'] % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        config['device']['distributed'] = False
        config['device']['is_main_process'] = True
        config['device']['gpu'] = 0
        return config

    config['device']['distributed'] = True

    torch.cuda.set_device(config['device']['gpu'])
    config['device']['dist_backend'] = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        config['device']['rank'], config['device']['dist_url']), flush=True)
    torch.distributed.init_process_group(backend=config['device']['dist_backend'], init_method=config['device']['dist_url'],
                                         world_size=config['device']['world_size'], rank=config['device']['rank'])
    torch.distributed.barrier()
    setup_for_distributed(config['device']['rank'] == 0)
    config['device']['is_main_process'] = is_main_process()
    return config

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
