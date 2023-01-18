#!/bin/sh

# ./rasa_usrun.sh -p RTX6000 --gpus=1 --time 8:00:00 /netscratch/akahmed/modular-adapter/rasa/slurm/rasa_experiment.sh rasa_experiment
# ./rasa_usrun.sh -p RTX6000 --gpus=1 --mem 60000 --cpus-per-gpu=1 --pty --time 08:00:00 /bin/bash

# export TORCH_HOME=/netscratch/akahmed/cache/torch
# export PIP_CACHE_DIR=/netscratch/akahmed/cache/pip

 # i get a weird error without the following line because somebody sets LOCAL_RANK=0 which confuses everything...
export LOCAL_RANK=-1 

cd /netscratch/akahmed/modular-adapter/rasa/
# pip3 install -U pip

pip3 install -r requirements.txt

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


if [ $1 = "rasa_experiment" ]; then
    python train_rasa.py --experiment rasa_experiment
elif [ $1 = "duration_experiment" ]; then
    python train_rasa.py --experiment duration_experiment
fi


