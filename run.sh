#!/bin/bash
export PYTHONPATH=$(pwd)
python configs/path_setup.py

torchrun --nproc_per_node=2 src/training/ddp_script.py
