import pandas as pd
import os
import numpy as np
import sys
import json
import random
import logging
import threading

# save all infos about training data in one place 
class DataContext():

    def __init__(self):

        self.task=None
        self.train_dataset=None
        self.valid_dataset=None
        self.test_dataset=None
        self.intent_labels = None
        self.intent_label_to_id = None
        self.domain_labels = None
        self.domain_label_to_id = None
        self.rasa_df = None
        self.rasa_df_json = None
        self.train_json = None
        self.test_json = None
        self.valid_json = None
        self.test_all_json = None
        self.train_json_path = None
        self.test_json_path = None
        self.valid_json_path = None
        self.test_all_json_path = None
        self.domains = None

    def get_labels(self):
        if self.task == "intent":
            return self.intent_labels, self.intent_label_to_id
        elif self.task == "domain":
            return self.domain_labels, self.domain_label_to_id
        else:
            raise Exception(f"task needs to be intent or domain but is {self.task}")


def load_rasa_dataset(args, task, domains, shuffle_train=True) -> DataContext:
    context = DataContext()

    context.domains = domains
    context.task = task
    f = os.path.join(args.dataset_path, args.dataset + ".csv")
    context.rasa_df = pd.read_csv(f)
    context.rasa_df = context.rasa_df.rename(columns={"Unnamed: 0": "id"})

    context.intent_labels = list(context.rasa_df.intent.unique())
    context.intent_label_ids = {}
    for l in context.rasa_df.intent.unique():
        context.intent_label_ids[l] = len(context.intent_label_ids)

    context.domain_labels = list(context.rasa_df.domain.unique())
    context.domain_label_ids = {}
    for l in context.rasa_df.domain.unique():
        context.domain_label_ids[l] = len(context.domain_label_ids)

    if context.domains is not None:
        context.train_dataset = context.rasa_df[context.rasa_df.domain.apply(lambda x : x in context.domains)]
        context.valid_dataset = context.rasa_df[context.rasa_df.domain.apply(lambda x : x in context.domains)]
        context.test_dataset = context.rasa_df[context.rasa_df.domain.apply(lambda x : x in context.domains)]
    
    # filter for dataset
    context.train_dataset = context.train_dataset[context.train_dataset.dataset=="train"]
    context.valid_dataset = context.valid_dataset[context.valid_dataset.dataset=="val"]
    context.test_dataset = context.test_dataset[context.test_dataset.dataset=="test"]
    context.test_dataset_all = context.rasa_df[context.rasa_df.dataset=="test"]

    num_workers=0

    context.train_json = create_nlu_json(args, context.train_dataset)
    context.train_json_path = f"rasa_dataset/{task}_{args.dataset}_{args.num_modules}_train.json"
    write_json(context.train_json, context.train_json_path)

    context.test_json = create_nlu_json(args, context.test_dataset)
    context.test_json_path = f"rasa_dataset/{task}_{args.dataset}_{args.num_modules}_test.json"
    write_json(context.test_json, context.test_json_path)

    context.valid_json  = create_nlu_json(args, context.valid_dataset)
    context.valid_json_path = f"rasa_dataset/{task}_{args.dataset}_{args.num_modules}_valid.json"
    write_json(context.valid_json, context.valid_json_path)

    context.test_all_json = create_nlu_json(args, context.test_dataset_all)
    context.test_all_json_path = f"rasa_dataset/{task}_{args.dataset}_{args.num_modules}_test_all.json"
    write_json(context.test_all_json, context.test_all_json_path)

    return context


# load the assignment of domains to dataset
# n_modules determines on how many modules to split the domains 
def load_domain_assignment(args):
    inputfile = os.path.join(args.dataset_path, args.dataset + "_domains.json")
    f = open(inputfile, "r")
    domains = json.load(f)
    f.close()

    random.shuffle(domains)

    domain_assignment = [[] for i in range(args.num_modules)]
    for i in range(len(domains)):
        domain_assignment[i%args.num_modules].append(domains[i])

    return domain_assignment


def create_nlu_json(args, df):
    common_examples = []
    intents = df["intent"].unique()
    for intent in intents:
        examples = []
        used_utterances = set()
        for ix, row in df[df["intent"] == intent].iterrows():
            if row["text"] in used_utterances:
                continue

            used_utterances.add(row["text"])
            if args.dataset == "atis" or args.dataset == "banking77":
                common_examples.append({
                    "text": row["text"],
                    "intent": "#"+intent,
                    "entities": []
                })
            else:                
                common_examples.append({
                    "text": row["text"],
                    "intent": "#"+intent,
                    "entities": [],
                    "domain": row["domain"]
                })
    json = {
        "rasa_nlu_data": {
            "common_examples": common_examples
        }
    }
    return json

def write_json(data, filename):
    json_str = json.dumps(data, indent=4)
    with open(filename, "w") as file:
        file.write(json_str)
    return