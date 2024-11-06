#!/bin/bash
export PYTHONPATH=$(pwd)
python configs/path_setup.py
python src/training/ddp_script.py
