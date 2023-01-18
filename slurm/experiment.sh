#!/bin/sh

# ./usrun.sh -p RTX6000 --gpus=1 --time 36:00:00 /netscratch/nehring/projects/modular-adapter/slurm/experiment.sh full_experiment

export TORCH_HOME=/netscratch/nehring/cache/torch
export PIP_CACHE_DIR=/netscratch/nehring/cache/pip
export PIP_DOWNLOAD_DIR=/netscratch/nehring/cache/pip
export HF_HOME=/netscratch/nehring/cache/huggingface

 # i get a weird error without the following line because somebody sets LOCAL_RANK=0 which confuses everything...
export LOCAL_RANK=-1 

cd /netscratch/nehring/projects/modular-adapter
pip install -r requirements.txt
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

python prepare_svm.py

if [ $1 = "bert_only" ]; then
    python train.py
elif [ $1 = "full_experiment" ]; then
    python train.py --experiment full_experiment
elif [ $1 = "search_hyperparameter" ]; then
    python train.py --experiment search_hyperparameter
elif [ $1 = "duration_experiment" ]; then
    python train.py --experiment duration_experiment
fi


