# Adapters for Resource-Efficient Deployment of NLU models

Source codes to the paper.

## How to install

In a new virtual environment with Python 3.6+, do

```
git clone ...
pip install -r requirements.txt
```

## How to run?

train.py is the main executable. Execute `python train.py -h` for a list of command line options. Here are some example calls:

```
train and evaluate the bert model on the hwu dataset
python train.py
```

```
train and evaluate bert+adapter model on the clinc dataset
python train.py --add_adapter --dataset clinc
```

```
train a model for intent detection only (non-modular scenario) with adapter and save the models
python train.py --add_adapter --save_model --num_modules 1
```

load this pretrained model from previous step to skip training and evaluate again

```
python train.py \
    --add_adapter \
    --skip_training \
    --pretrained_model_path logs/2022-02-17T10-17-06_\,add_adapter\=True\,single_experiment/ \
    --num_modules 1
```

The parameter `--experiment` determines which experiment to run. It defaults to `single_experiment`. Possible values are

* `single experiment` Run a single training and evaluation run, e.g. evaluate on model on one dataset
* `full experiment` Run all models on all datasets with varying number of modules
* `model_size_experiment` Procuce model size in MB
* `duration_experiment` Measure durations.

The skripts in the `eval` folder produces the numbers, chart and tables of the paper.

## Slurm

We run the experiments on a Slurm cluster. The folder `slurm` shows the example skript how to execute the experiment on a slurm cluster.

## License

The codes are released under the Apache License Version 2.0. 