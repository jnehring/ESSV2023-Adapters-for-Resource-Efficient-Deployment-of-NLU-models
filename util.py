
import argparse
import logging
from datetime import datetime
import os
import random
import torch
import numpy as np
import sys
import pandas as pd

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_path", default="data", type=str)
    parser.add_argument("--dataset", default="hwu", type=str)
    parser.add_argument("--token_vocab_path", type=str)
    parser.add_argument("--log_folder", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--adapter_train_batch_size", type=int, default=256)
    parser.add_argument("--adapter_test_batch_size", type=int, default=256)
    parser.add_argument("--non_adapter_train_batch_size", type=int, default=256)
    parser.add_argument("--non_adapter_test_batch_size", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--num_modules", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--add_adapter", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--adapter_learning_rate", default=0.005, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--non_adapter_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--device", default=0, type=int, help="GPU device #")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default="single_experiment", choices=["single_experiment", "full_experiment", "search_hyperparameter", "duration_experiment"])
    return parser.parse_args()

# create the output folder
def make_log_folder(args, name):
    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    my_date = datetime.now()

    folder_name=my_date.strftime('%Y-%m-%dT%H-%M-%S') + "_" + name

    if len(args.experiment_name) > 0:
        folder_name += "_" + args.experiment_name

    log_folder=os.path.join(args.log_folder, folder_name)
    os.mkdir(log_folder)
    return log_folder

# write a single store to the results logfile
results_df=None
def write_results(log_dir, key, value):

    logging.info(f"write result row key={key}, value={value}")
    results_file=os.path.join(log_dir, "results.csv")

    global results_df
    if results_df is None:
        results_df=pd.DataFrame(columns=["key", "value"])
    
    data=[key, value]
    results_df.loc[len(results_df)]=data

    results_df.to_csv(results_file)

# log to file and console
def create_logger(log_dir):
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

# set all random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_experiments(args, experiment_name):
    log_dir=make_log_folder(args, experiment_name)
    logging.info(log_dir)
    create_logger(log_dir)
    set_seed(args.seed)

    command=" ".join(sys.argv)
    logging.info('''
 __  __           _       _                          _             _            
|  \/  |         | |     | |                /\      | |           | |           
| \  / | ___   __| |_   _| | __ _ _ __     /  \   __| | __ _ _ __ | |_ ___ _ __ 
| |\/| |/ _ \ / _` | | | | |/ _` | '__|   / /\ \ / _` |/ _` | '_ \| __/ _ \ '__|
| |  | | (_) | (_| | |_| | | (_| | |     / ____ \ (_| | (_| | |_) | ||  __/ |   
|_|  |_|\___/ \__,_|\__,_|_|\__,_|_|    /_/    \_\__,_|\__,_| .__/ \__\___|_|   
                                                            | |                 
                                                            |_|                 ''')
    logging.info("start command: " + command)
    logging.info(f"experiment name {experiment_name}")
    logging.info(args)
    return log_dir

