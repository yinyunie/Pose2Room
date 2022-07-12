#!/bin/sh
conda activate p2rnet
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=$((RANDOM + 9000)) main.py --config configs/config_files/p2rnet_train.yaml --mode train
