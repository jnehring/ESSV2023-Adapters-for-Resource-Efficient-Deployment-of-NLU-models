from transformers import BertModelWithHeads, BertConfig, BertForSequenceClassification
import numpy as np
from transformers import BertModel, BertConfig, TrainingArguments, AdapterTrainer, Trainer, EvalPrediction
from transformers import RobertaConfig, RobertaModelWithHeads, EarlyStoppingCallback, default_data_collator
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelWithHeads

import argparse
from dataloader import load_dataset, load_domain_assignment
from util import init_experiments, read_args, write_results
import logging
import sys
import os
import time
import torch
import shutil
from sklearn.metrics import accuracy_score, f1_score
import json

import train_svm

# load a normal transformer without adapter
def load_base_model(args, num_labels, sub_dir):

    if args.add_adapter:

        if args.model_name_or_path == "bert-base-uncased":
            model = BertModelWithHeads.from_pretrained(args.model_name_or_path)
            
            if args.pretrained_model_path is not None:
                adapter_folder=os.path.join(args.pretrained_model_path, sub_dir, "modular_chatbot")
                adapter_name = model.load_adapter(adapter_folder)
                model.set_active_adapters(adapter_name)
        elif args.model_name_or_path == "distilbert-base-uncased":
            model = AutoModelWithHeads.from_pretrained(args.model_name_or_path)
        else:
            raise Exception(f"unknown model {args.model_name_or_path}" )
    else:
        if args.pretrained_model_path is not None:
            model=BertForSequenceClassification.from_pretrained(os.path.join(args.pretrained_model_path, sub_dir))
        elif args.model_name_or_path == "bert-base-uncased":
            config = BertConfig.from_pretrained(
                args.model_name_or_path,
                num_labels=num_labels,
            )
            model = BertForSequenceClassification(config)
        elif args.model_name_or_path == "distilbert-base-uncased":
            model = DistilBertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        else:
            raise Exception(f"unknown model \"{args.model_name_or_path}\"")
    num_p=sum(p.numel() for p in model.parameters())
    logging.info(f"loaded model with {num_p:,} parameters.")
    return model

# load a trainer, either with adapter or without
def load_trainer(args, model, output_dir, data_context):

    train_batch_size=args.adapter_train_batch_size if args.add_adapter else args.non_adapter_train_batch_size
    test_batch_size=args.adapter_test_batch_size if args.add_adapter else args.non_adapter_test_batch_size
    learning_rate=args.adapter_learning_rate if args.add_adapter else args.non_adapter_learning_rate

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        logging_steps=200,
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=None,
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    trainer_args={
        "model": model,
        "args": training_args,
        "train_dataset": data_context.train_dataset,
        "eval_dataset": data_context.valid_dataset,
        "compute_metrics": compute_accuracy,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.patience, early_stopping_threshold=0.002)]
    }

    if args.add_adapter:
        trainer = AdapterTrainer(**trainer_args)
    else:
        trainer=Trainer(**trainer_args)
    return trainer

# load base model, add uninitialized adapter (if args.adapter=True), train it and save it
# log_dir is the projects log_dir
# output dir is the last part of the output directory, e.g. "intent_detector_0"
def train(args, data_context, task, log_dir, output_dir):

    logging.info(f"train model with add_adapter={args.add_adapter}" )
    labels, label_to_id=data_context.get_labels()

    model=load_base_model(args, len(labels), output_dir)

    if args.add_adapter and args.pretrained_model_path is None:
        num_p_before=sum(p.numel() for p in model.parameters())
        model.add_adapter("modular_chatbot")
        model.add_classification_head(
            "modular_chatbot",
            num_labels=len(labels),
            id2label=label_to_id
        )
        model.train_adapter("modular_chatbot")
        num_p=sum(p.numel() for p in model.parameters())
        delta=num_p-num_p_before
        logging.info(f"added adapter, number of parameters is now {num_p}. This is {delta} new parameters through the adapter.")

    trainer=load_trainer(args, model, os.path.join(log_dir, output_dir), data_context)

    if args.skip_training:
        logging.info("skip training because args.skip_training=True")
        return model, trainer
    
    starttime=time.time()
    trainer.train()
    duration=(time.time()-starttime)

    logging.info(f"training finished in {(duration/60):.2f} minutes")

    if args.save_models:
        if args.add_adapter:
            model.save_all_adapters(os.path.join(log_dir, output_dir), with_head=True)
        else:
            model.save_pretrained(os.path.join(log_dir, output_dir))

    # cleanup all checkpoint directories
    for f in os.listdir(os.path.join(log_dir, output_dir)):
        needle="checkpoint-"
        if f[0:len(needle)]==needle:
            shutil.rmtree(os.path.join(log_dir, output_dir, f))


    return model, trainer, duration

# load a model from file
def load_model_from_file(args, data_context, adapter_path=None):
    labels, label_to_id=data_context.get_labels()
    model=load_base_model(args.model_name_or_path, len(labels), args.add_adapter)
    if args.add_adapter and adapter_path is not None:
        adapter_name = model.load_adapter(adapter_path)
        model.set_active_adapters(adapter_name)
    return model

# run a single experiment that trains a module selector (if args.num_modules>1) and args.num_modules intent detectors
# test_on_valid runs the evaluation on the validation set instead of on the test set, which we use for hyperparameter search
def single_experiment(args, log_dir, log_praefix="", test_on_valid=False):

    experiment_start_time=time.time()

    domain_assignment=load_domain_assignment(args)

    def prediction_helper(results, mapping):
        y_pred=[p.argmax() for p in results.predictions]
        y_pred=[mapping[x] for x in y_pred]
        y_true=[mapping[x] for x in results.label_ids]

        acc=accuracy_score(y_true, y_pred)
        f1=f1_score(y_true, y_pred, average="micro")
        return y_true, y_pred, acc, f1

    # train module selector
    if args.num_modules <=1:
        logging.info(f"skip training of module selector because num_modules={args.num_modules}")
    else:
        logging.info("train module selector")
        task="domain"
        domains_flat=[item for sublist in domain_assignment for item in sublist]
        data_context = load_dataset(args, task, domains_flat)

        # run test set and compute acc_ms and f1_ms
        model, trainer, duration=train(args, data_context, task, log_dir, "module_selector")
        write_results(log_dir, f"{log_praefix}train_duration", duration)
        write_results(log_dir, f"{log_praefix}train_num_samples", len(data_context.train_dataset))
        model=model.to(model.device)

        test_set=data_context.valid_dataset if test_on_valid else data_context.test_dataset
        starttime=time.time()
        result_ms=trainer.predict(test_set)
        duration=(time.time()-starttime)
        y_true_ms, y_pred_ms, acc, f1=prediction_helper(result_ms, data_context.domain_labels)
        write_results(log_dir, f"{log_praefix}acc_ms", acc)
        write_results(log_dir, f"{log_praefix}f1_ms", f1)
        write_results(log_dir, f"{log_praefix}predict_duration", duration)
        write_results(log_dir, f"{log_praefix}predict_num_samples", len(test_set))

    # train intent detector
    all_y_true_id=None
    all_y_pred_id=[]
    for i_module in range(args.num_modules):
        logging.info(f"train intent detector {i_module+1}/{args.num_modules}")
        task="intent"
        data_context = load_dataset(args, task, domain_assignment[i_module])
        model, trainer, duration=train(args, data_context, task, log_dir, f"intent_detector_{i_module}")
        write_results(log_dir, f"{log_praefix}train_duration", duration)
        write_results(log_dir, f"{log_praefix}train_num_samples", len(data_context.train_dataset))
        model=model.to(model.device)

        # run in domain test set and compute acc_id and f1_id for indomain
        test_set=data_context.valid_dataset if test_on_valid else data_context.test_dataset
        starttime=time.time()
        result_id=trainer.predict(test_set)
        duration=(time.time()-starttime)
        y_true_id, y_pred_id, acc, f1=prediction_helper(result_id, data_context.intent_labels)
        write_results(log_dir, f"{log_praefix}acc_id_{i_module}", acc)
        write_results(log_dir, f"{log_praefix}f1_id_{i_module}", f1)
        write_results(log_dir, f"{log_praefix}predict_duration", duration)
        write_results(log_dir, f"{log_praefix}predict_num_samples", len(test_set))

        if args.num_modules>1:
            # run the complete test set
            result_id_all=trainer.predict(data_context.test_dataset_all)
            y_true_id, y_pred_id, acc, f1=prediction_helper(result_id_all, data_context.intent_labels)
            all_y_pred_id.append(y_pred_id)
            all_y_true_id=y_true_id

    if args.num_modules==1:
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
                domain_id[domain]=i

        y_pred=[]
        for i in range(len(y_true_ms)):
            module=domain_id[y_pred_ms[i]]
            y_pred.append(all_y_pred_id[module][i])

        acc=accuracy_score(all_y_true_id, y_pred)
        f1=f1_score(all_y_true_id, y_pred, average="micro")
        write_results(log_dir, f" {log_praefix}acc_id_all", acc)
        write_results(log_dir, f"{log_praefix}f1_id_all", f1)

    duration=(time.time()-experiment_start_time)
    write_results(log_dir, f" {log_praefix}total_duration", duration)

# perform a hyperparameter search for optimal learning rate and batch size on hwu64 dataset for models bert and bert+adapter
def search_hyperparameter(args, log_dir):
    args.add_adapter=False
    for batch_size in [8,16,32,64,128,256]:
        for learning_rate in [1e-5, 3e-5, 5e-5]:
            args.non_adapter_train_batch_size=batch_size
            args.non_adapter_learning_rate=learning_rate
            log_prefix=f"model=bert,learning_rate={learning_rate},batch_size={batch_size},"
            single_experiment(args, log_dir, log_praefix=log_prefix, test_on_valid=True)

    args.add_adapter=True
    for batch_size in [8,16,32,64,128,256,512]:
        for learning_rate in [5e-3, 1e-4, 3e-4, 5e-4]:
            args.adapter_train_batch_size=batch_size
            args.non_adapter_learning_rate=learning_rate
            log_prefix=f"model=bert+adapter,learning_rate={learning_rate},batch_size={batch_size},"
            single_experiment(args, log_dir, log_praefix=log_prefix, test_on_valid=True)

def full_experiment(args, log_dir):

    for dataset in ["atis", "banking77", "hwu", "clinc", "hwu_orig"]:
        if dataset == "atis" or dataset == "banking77":
            modules=[1]
        else:
            modules=[1,3,10]
        modules=[1]
        for num_modules in modules:
            for model in "svm", "bert-base-uncased", "distilbert-base-uncased":
                for add_adapter in [True, False]:

                    if model == "svm" and add_adapter:
                        continue

                    args.dataset=dataset
                    args.add_adapter=add_adapter 
                    args.num_modules=num_modules
                    args.model_name_or_path=model

                    model_name=model
                    if args.model_name_or_path == "bert-base-uncased" and add_adapter:
                        model_name = "bert+adapter"
                    if args.model_name_or_path == "distilbert-base-uncased" and add_adapter:
                        model_name = "distilbert+adapter"

                    log_praefix=f"dataset={args.dataset}"
                    log_praefix+=f",add_adapter={args.add_adapter}"
                    log_praefix+=f",num_modules={args.num_modules}"
                    log_praefix+=f",model={model_name}"
                    log_praefix+=","

                    if model == "svm":
                        train_svm.svm_single_experiment(args, log_dir, log_praefix=log_praefix)
                    else:
                        single_experiment(args, log_dir, log_praefix=log_praefix)

def duration_experiment(args, log_dir):
    for dataset in ["hwu", "clinc", "hwu_orig", "atis", "banking77"]:
        args.dataset=dataset
        args.num_modules=1
        for i in range(0, 10):
            for model in "svm", "bert-base-uncased", "distilbert-base-uncased":
                for add_adapter in [True, False]:

                    if model == "svm" and add_adapter:
                        continue

                    args.add_adapter=add_adapter 
                    args.model_name_or_path=model

                    model_name=model
                    if args.model_name_or_path == "bert-base-uncased" and add_adapter:
                        model_name = "bert+adapter"
                    if args.model_name_or_path == "distilbert-base-uncased" and add_adapter:
                        model_name = "distilbert+adapter"

                    log_praefix=f"model={model_name},i={i},dataset={dataset},"

                    if model == "svm":
                        train_svm.svm_single_experiment(args, log_dir, log_praefix=log_praefix)
                    else:
                        single_experiment(args, log_dir, log_praefix=log_praefix)

def main():
    try:
        starttime=time.time()
        args=read_args()
        
        if args.experiment == "single_experiment":
            name=f"dataset={args.dataset}"
            name=f"..num_modules={args.num_modules}"
            name=f"..add_adapter={args.add_adapter}"
            name=f"..model={args.model_name_or_path}"
            if args.subsample>0:
                name +="..subsample"  
            log_dir=init_experiments(args, name + "..single_experiment")
            single_experiment(args, log_dir)
        elif args.experiment == "full_experiment":
            log_dir=init_experiments(args, "full_experiment")
            full_experiment(args, log_dir)
        elif args.experiment == "search_hyperparameter":
            log_dir=init_experiments(args, "search_hyperparameter")
            search_hyperparameter(args, log_dir)
        elif args.experiment == "duration_experiment":
            log_dir=init_experiments(args, "duration_experiment")
            duration_experiment(args, log_dir)

        duration=(time.time()-starttime)/60
        logging.info(f"finished in {duration:.2f} minutes")

    except Exception as e:
        logging.exception(e)

if __name__ == '__main__':
    main()

