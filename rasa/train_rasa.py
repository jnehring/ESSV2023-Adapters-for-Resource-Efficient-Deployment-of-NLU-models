import numpy as np
import argparse
import logging
import sys
import os
import time
import shutil
import json
import tensorflow as tf
import pandas as pd
import statistics
from rasa.nlu.model import Trainer
from rasa.shared.nlu.training_data.loading import load_data
from rasa.nlu import config
from rasa.nlu.model import Interpreter
from dataloader_rasa import load_rasa_dataset, load_domain_assignment
from util_rasa import init_experiments, read_args, write_results
from sklearn.metrics import accuracy_score, f1_score
from rasa.cli.utils import get_validated_path
from rasa.model import get_local_model, get_model_subdirectories
from sklearn.metrics import classification_report


def train_rasa(args, data_context, task, log_dir, output_dir):
    logging.info(f"train model with add_adapter={args.add_adapter}" )
    labels, label_to_id = data_context.get_labels()

    parent_path = "/netscratch/akahmed/modular-adapter/rasa"
    data_path = os.path.join(parent_path, "rasa_dataset", f"{task}_{args.dataset}_{args.num_modules}_train.json")

    models_path = os.path.join(parent_path, "models")
    if not os.path.isdir(models_path):
        os.mkdir(models_path)

    configs = "/netscratch/akahmed/modular-adapter/rasa/config_spacy.yaml"
    training_data = load_data(data_path)
    trainer = Trainer(config.load(configs))
    if task == "domain":
        for k, v in enumerate(training_data.training_examples):
            training_data.training_examples[k].data["intent"] = training_data.training_examples[k].data["domain"]

    
    starttime = time.time()
    trainer.train(training_data)
    duration = (time.time() - starttime)
    model_directory = trainer.persist(models_path, fixed_model_name = f"{task}_{args.dataset}_{args.num_modules}_train")

    logging.info(f"training finished in {(duration/60):.2f} minutes")

    return model_directory, trainer, duration


def get_prediction_scores(args, model, test_path_dir, task):
    f = open(test_path_dir, "r", encoding = "utf-8")
    test_set = json.load(f)

    msg = []
    intent = []
    domain = []
    for i in test_set["rasa_nlu_data"]["common_examples"]:
        m = ""
        if i["text"]:
            m = i["text"]

        intnt = ""
        if i["intent"]:
            intnt = i["intent"]

        dom = ""
        if args.dataset in ["atis", "banking77"]:
            dom = args.dataset
        else:
            if i["domain"]:
                dom = i["domain"]

        msg.append(m)
        intent.append(intnt)
        domain.append(dom)
    if task == "domain":
        test_df = {
            "text": msg,
            "intent": domain
        }
    else:
        test_df = {
            "text": msg,
            "intent": intent
        }

    test_df = pd.DataFrame(test_df)

    def add_predictions(dataf, nlu):
        pred_blob = [nlu.parse(t)['intent'] for t in dataf["text"]]
        return (dataf
                [['text', 'intent']]
                .assign(pred_intent=[p['name'] for p in pred_blob])
                .assign(pred_confidence=[p['confidence'] for p in pred_blob]))

    # Turn this list into a dataframe and add predictions using the nlu-interpreter
    df_intents = test_df.pipe(add_predictions, nlu=model)
    df_summary = (df_intents
        .groupby('pred_intent')
        .agg(n=('pred_confidence', 'size'),
            mean_conf=('pred_confidence', 'mean')))
        
    report = classification_report(y_true=df_intents['intent'], y_pred=df_intents['pred_intent'], output_dict = True)
    acc = accuracy_score(df_intents['intent'], df_intents['pred_intent'])
    f1 = report["weighted avg"]["f1-score"]

    return df_intents['intent'], df_intents['pred_intent'], acc, f1, len(test_set["rasa_nlu_data"]["common_examples"])


# run a single experiment that trains a module selector (if args.num_modules>1) and args.num_modules intent detectors
# test_on_valid runs the evaluation on the validation set instead of on the test set
def rasa_single_experiment(args, log_dir, log_praefix="", test_on_valid=False):
    experiment_start_time = time.time()
    domain_assignment = load_domain_assignment(args)

    parent_path = "/netscratch/akahmed/modular-adapter/rasa"
    rasa_data_path = os.path.join(parent_path, "rasa_dataset")
    if not os.path.isdir(rasa_data_path):
        os.mkdir(rasa_data_path)

    # train module selector
    if args.num_modules <= 1:
        logging.info(f"skip training of module selector because num_modules={args.num_modules}")
    else:
        logging.info("train module selector")
        task = "domain"
        domains_flat = [item for sublist in domain_assignment for item in sublist]
        data_context = load_rasa_dataset(args, task, domains_flat)
        print(data_context.train_dataset)
        # run test set and compute acc_ms and f1_ms
        model_dir, trainer, duration = train_rasa(args, data_context, task, log_dir, "module_selector")
        write_results(log_dir, f"{log_praefix}train_duration", duration)
        write_results(log_dir, f"{log_praefix}train_num_samples", len(data_context.train_dataset))

        ms_model = Interpreter.load(model_dir)
        test_set_path = data_context.valid_json_path if test_on_valid else data_context.test_json_path
        starttime = time.time()
        y_true_ms, y_pred_ms, acc, f1, test_set_len = get_prediction_scores(args, ms_model, test_set_path, task)
        duration = (time.time()-starttime)
        
        write_results(log_dir, f"{log_praefix}acc_ms", acc)
        write_results(log_dir, f"{log_praefix}f1_ms", f1)
        write_results(log_dir, f"{log_praefix}predict_duration", duration)
        write_results(log_dir, f"{log_praefix}predict_num_samples", test_set_len)

    # train intent detector
    all_y_true_id = None
    all_y_pred_id = []
    for i_module in range(args.num_modules):
        logging.info(f"train intent detector {i_module + 1}/{args.num_modules}")
        task = "intent"
        data_context = load_rasa_dataset(args, task, domain_assignment[i_module])
        model_dir, trainer, duration = train_rasa(args, data_context, task, log_dir, f"intent_detector_{i_module}")
        write_results(log_dir, f"{log_praefix}train_duration", duration)
        write_results(log_dir, f"{log_praefix}train_num_samples", len(data_context.train_dataset))

        # run in domain test set and compute acc_id and f1_id for indomain
        model_id = Interpreter.load(model_dir)
        test_set_path_id = data_context.valid_json_path if test_on_valid else data_context.test_json_path
        starttime = time.time()
        y_true_id, y_pred_id, acc, f1, test_set_len = get_prediction_scores(args, model_id, test_set_path_id, task)
        duration=(time.time() - starttime)

        write_results(log_dir, f"{log_praefix}acc_id_{i_module}", acc)
        write_results(log_dir, f"{log_praefix}f1_id_{i_module}", f1)
        write_results(log_dir, f"{log_praefix}predict_duration", duration)
        write_results(log_dir, f"{log_praefix}predict_num_samples", test_set_len)

        if args.num_modules>1:
            # run the complete test set
            y_true_id, y_pred_id, acc, f1, test_set_len = get_prediction_scores(args, model_id, data_context.test_all_json_path, task)
            all_y_pred_id.append(y_pred_id)
            all_y_true_id = y_true_id

    if args.num_modules == 1:
        write_results(log_dir, f"{log_praefix}acc_id_all", acc)
        write_results(log_dir, f"{log_praefix}f1_id_all", f1)
    else:
        # compute final predictions of intent detection
        assert len(y_true_ms) == len(y_pred_ms)
        assert len(y_true_ms) == len(all_y_true_id)
        for i in range(len(all_y_pred_id)):
            assert len(all_y_pred_id[i]) == len(y_pred_ms)

        domain_id={}
        for i in range(len(domain_assignment)):
            for domain in domain_assignment[i]:
                domain_id[domain] = i

        y_pred = []
        for i in range(len(y_true_ms)):
            module = domain_id[y_pred_ms[i]]
            y_pred.append(all_y_pred_id[module][i])

        acc = accuracy_score(all_y_true_id, y_pred)
        f1 = f1_score(all_y_true_id, y_pred, average = "micro")
        write_results(log_dir, f" {log_praefix}acc_id_all", acc)
        write_results(log_dir, f"{log_praefix}f1_id_all", f1)

    duration = (time.time() - experiment_start_time)
    write_results(log_dir, f" {log_praefix}total_duration", duration)

    return


def rasa_experiment(args, log_dir):
    for dataset in ["atis", "banking77", "hwu", "clinc", "hwu_orig"]:
        for num_modules in [1,3,10]:
            if dataset in ["atis", "banking77"] and num_modules>1:
                continue

            args.dataset = dataset
            args.add_adapter = False
            args.num_modules = num_modules

            log_praefix = f"dataset={args.dataset}"
            log_praefix += f",add_adapter={args.add_adapter}"
            log_praefix += f",num_modules={args.num_modules}"
            log_praefix += ","
            rasa_single_experiment(args, log_dir, log_praefix = log_praefix)
    return


def duration_experiment(args, log_dir):
    for dataset in ["atis", "banking77"]:#, "hwu", "clinc", "hwu_orig"]:
        args.dataset = dataset
        args.num_modules = 1
        for i in range(0, 10):
            args.add_adapter = False
            log_praefix = f"model=rasa, i={i}, dataset={dataset},"
            rasa_single_experiment(args, log_dir, log_praefix = log_praefix)
    return


def calculate_rasa_sps(result_path = "/netscratch/akahmed/modular-adapter/rasa/logs/2022-08-17T13-43-35_duration_experiment/results.csv"):
    df = pd.read_csv(result_path)
    dic = {
        "hwu_dur": [], 
        "hwu_samples": 8954,
        "atis_dur":[],
        "atis_samples": 4480,
        "banking77_dur":[],
        "banking77_samples": 9003,
        "clinc_dur": [],
        "clinc_samples": 15000,
        "hwu_orig_dur": [],
        "hwu_orig_samples": 19879,
        "hwu_inf_samples": 1076,
        "hwu_inf_dur": [],
        "clinc_inf_samples": 4500,
        "clinc_inf_dur": [],
        "hwu_orig_inf_samples": 2510,
        "hwu_orig_inf_dur": [],
        "atis_inf_samples": 850,
        "atis_inf_dur": [],
        "banking77_inf_samples": 3080,
        "banking77_inf_dur":[]

    }

    indx = 0
    inf_indx = 4
    for i in range(3):
        for j in range(10):
            if indx <90 :
                # col = "hwu_dur"
                col = "atis_dur"
            elif indx < 180:
                # col = "clinc_dur"
                col = "banking77_dur"
            else:
                col = "hwu_orig_dur"
            if indx >= 180:
                break
            dic[col].append(df["value"][indx])
            indx += 9

            if inf_indx < 94:
                # col = "hwu_inf_dur"
                col = "atis_inf_dur"
            elif inf_indx < 184:
                # col = "clinc_inf_dur"
                col = "banking77_inf_dur"
            else:
                col = "hwu_orig_inf_dur"

            dic[col].append(df["value"][inf_indx]) 
            inf_indx += 9

    # dic = pd.DataFrame(dic)
    # print(dic.info())
    hwu_sps = round((dic["atis_samples"] / (sum(dic["atis_dur"]) / len(dic["atis_dur"]))), 2)
    hwu_stdev =  round(statistics.stdev(dic["atis_dur"]),2)

    clinc_sps = round((dic["banking77_samples"] / (sum(dic["banking77_dur"]) / len(dic["banking77_dur"]))), 2)
    clinc_stdev = round(statistics.stdev(dic["banking77_dur"]),2)

    # hwu_orig_sps = round((dic["hwu_orig_samples"] / (sum(dic["hwu_orig_dur"]) / len(dic["hwu_orig_dur"]))), 2)
    # hwu_orig_stdev =  round(statistics.stdev(dic["hwu_orig_dur"]), 2)
    
    # avg_rasa = round((hwu_sps + hwu_orig_sps + clinc_sps) / 3, 2)
    avg_rasa = round((hwu_sps + clinc_sps) / 2, 2)
    # rasa_dur_ls = dic["hwu_dur"] + dic["clinc_dur"] + dic["hwu_orig_dur"]
    rasa_dur_ls =  dic["atis_dur"] + dic["banking77_dur"] 
    avg_rasa_stdev = round(statistics.stdev(rasa_dur_ls) / len(rasa_dur_ls), 2)

    print(f"Avg SPS of Atis: {hwu_sps}({hwu_stdev})")
    print(f"Avg SPS of Banking77: {clinc_sps}({clinc_stdev})")
    # print(f"Avg SPS of HWU64-DialoGLUE: {hwu_orig_sps}({hwu_orig_stdev})")
    print(f"Avg SPS of Rasa: {avg_rasa}({avg_rasa_stdev})")

    hwu_inf_sps = round((dic["atis_inf_samples"]/(sum(dic["atis_inf_dur"])/len(dic["atis_inf_dur"]))),2)
    hwu_inf_stdev =  round(statistics.stdev(dic["atis_inf_dur"]),2)

    clinc_inf_sps = round((dic["banking77_inf_samples"]/(sum(dic["banking77_inf_dur"])/len(dic["banking77_inf_dur"]))),2)
    clinc_inf_stdev = round(statistics.stdev(dic["banking77_inf_dur"]),2)

    # hwu_orig_inf_sps = round((dic["hwu_orig_inf_samples"]/(sum(dic["hwu_orig_inf_dur"])/len(dic["hwu_orig_inf_dur"]))),2)
    # hwu_orig_inf_stdev =  round(statistics.stdev(dic["hwu_orig_inf_dur"]),2)
    
    # avg_inf_rasa = round((hwu_inf_sps + hwu_orig_inf_sps + clinc_inf_sps) / 3, 2)
    avg_inf_rasa = round((hwu_inf_sps + clinc_inf_sps) / 2, 2)
    rasa_inf_dur_ls = dic["atis_inf_dur"] + dic["banking77_inf_dur"] 
    avg_rasa_inf_stdev = round(statistics.stdev(rasa_inf_dur_ls) / len(rasa_inf_dur_ls), 2)

    print(f"Avg inf_SPS of Atis: {hwu_inf_sps}({hwu_inf_stdev})")
    print(f"Avg inf_SPS of Banking77: {clinc_inf_sps}({clinc_inf_stdev})")
    # print(f"Avg inf_SPS of HWU64-DialoGLUE: {hwu_orig_inf_sps}({hwu_orig_inf_stdev})")
    print(f"Avg inf_SPS of Rasa: {avg_inf_rasa}({avg_rasa_inf_stdev})")

    # df = pd.DataFrame({
    #     "Model": ["Rasa", "Rasa", "Rasa", "average Rasa"],
    #     "Dataset": ["CLINC150", "HWU64", "HWU64-DialoGLUE", ""],
    #     "SPS train": [f"{clinc_sps}({clinc_stdev})", f"{hwu_sps}({hwu_stdev})", f"{hwu_orig_sps}({hwu_orig_stdev})", f"{avg_rasa}({avg_rasa_stdev})"],
    #     "SPS inference": [f" {clinc_inf_sps}({clinc_inf_stdev})", f"{hwu_inf_sps}({hwu_inf_stdev})", f"{hwu_orig_inf_sps}({hwu_orig_inf_stdev})", f"{avg_inf_rasa}({avg_rasa_inf_stdev})"]
    # })
    # print(df.to_latex(index=False))

    return


def main():
    try:
        starttime = time.time()
        args=read_args()
        if args.experiment == "rasa_experiment":
            log_dir = init_experiments(args, "rasa_experiment")
            rasa_experiment(args, log_dir)
        elif args.experiment == "duration_experiment":
            log_dir = init_experiments(args, "duration_experiment")
            duration_experiment(args, log_dir)
            # calculate_rasa_sps() #need to send path for result directory from log folder

        duration = (time.time() - starttime) / 60
        logging.info(f"finished in {duration:.2f} minutes")
            
    except Exception as e:
        logging.exception(e)
    return 


if __name__ == '__main__':
    main()
    # calculate_rasa_sps()
