# Adapters for Resource-Efficient Deployment of NLU models

Modern Transformer-based language models such as BERT are huge and, therefore, expensive to deploy in practical applications. In environments such as commercial chatbot-as-a-service platforms that deploy many NLP models in parallel, less powerful models with a smaller number of parameters are an alternative to transformers to keep deployment costs down, at the cost of lower accuracy values. This paper compares different models for Intent Detection concerning their memory footprint, quality of Intent Detection, and processing speed. Many task-specific Adapters can share one large transformer model with the Adapter framework. The deployment of 100 NLU models requires 1 GB of memory for the proposed BERT+Adapter architecture, compared to 41.78 GB for a BERT-only architecture.

This repository contains the source codes for the paper [Nehring, J., Feldhus, N., Ahmed, A. (2023): Adapters for Resource-Efficient Deployment of NLU models](https://github.com/jnehring/ESSV2023-Adapters-for-Resource-Efficient-Deployment-of-NLU-models/blob/master/ESSV2023__Adapters_for_Resource_Efficient_Deployment_of_NLU_models.pdf)

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
