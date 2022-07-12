#!/bin/sh

conda activate p2rnet
python main.py --config ./configs/config_files/p2rnet_test.yaml --mode test
