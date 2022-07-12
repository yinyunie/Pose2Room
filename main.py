#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
import os
import argparse
from configs.config_utils import CONFIG, read_to_dict

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Pose2Room.')
    parser.add_argument('--config', type=str, default='configs/config_files/p2rnet_train.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo', help='Please specify the demo path.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # initialize devices
    config = read_to_dict(args.config)
    # initiate device environments
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']['gpu_ids']
    from net_utils.utils import initiate_environment, get_sha
    config = initiate_environment(config)

    # initialize config
    cfg = CONFIG(args, config)
    cfg.update_config(args.__dict__)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string("git:\n  {}\n".format(get_sha()))
    cfg.log_string(cfg.config)
    cfg.write_config()

    '''Run'''
    if cfg.config['mode'] == 'train':
        import train
        train.run(cfg)
    if cfg.config['mode'] == 'test':
        import test
        test.run(cfg)
    if cfg.config['mode'] == 'demo':
        import demo
        demo.run(cfg)

