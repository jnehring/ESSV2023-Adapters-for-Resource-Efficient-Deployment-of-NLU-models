from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
import sys
import json
import random
import torch
import logging
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, DistilBertTokenizer
import threading
from transformers import RobertaTokenizer

# save all infos about training data in one place 
class DataContext():

    def __init__(self):

        self.task=None
        self.tokenizer=None
        self.train_dataset=None
        self.valid_dataset=None
        self.test_dataset=None
        self.test_dataset_all=None
        self.train_dataloader=None
        self.valid_dataloader=None
        self.test_dataloader=None
        self.intent_labels=None
        self.intent_label_to_id=None
        self.domain_labels=None
        self.domain_label_to_id=None
        self.df=None
        self.domains=None

    def get_labels(self):
        if self.task == "intent":
            return self.intent_labels, self.intent_label_to_id
        elif self.task == "domain":
            return self.domain_labels, self.domain_label_to_id
        else:
            raise Exception(f"task needs to be intent or domain but is {self.task}")

# abstraction for the tokenization because BertTokenizer and BertWordPieceTokenizer work different
def tokenize(args, utterance, tokenizer):
    if type(tokenizer) == BertTokenizer:
        encoded = tokenizer(utterance, padding="max_length", max_length=args.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            encoded[key]=torch.LongTensor((encoded[key]))
        return encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]
    elif type(tokenizer) == DistilBertTokenizer:
        encoded = tokenizer(utterance, padding="max_length", max_length=args.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask"]:
            encoded[key]=torch.LongTensor((encoded[key]))
        return encoded["input_ids"], encoded["attention_mask"]
    else:
        raise Exception("unknown tokenizer")
        

# domains is a filter which domains to use. set to None to use all domains
# dataset can be either train, test or valid
# use subsample to speed up processing by subsampling.
def load_dataset(args, task, domains, shuffle_train=True) -> DataContext:
    
    context=DataContext()

    context.domains=domains
    context.task=task

    if args.model_name_or_path=="distilbert-base-uncased":
         context.tokenizer = DistilBertTokenizer.from_pretrained(args.model_name_or_path, model_max_length=args.max_seq_length)
    else:
        context.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=args.max_seq_length)

    f=os.path.join(args.dataset_path, args.dataset + ".csv")
    context.df=pd.read_csv(f)
    context.df=context.df.rename(columns={"Unnamed: 0": "id"})

    context.intent_labels=list(context.df.intent.unique())
    context.intent_label_ids={}
    for l in context.df.intent.unique():
        context.intent_label_ids[l]=len(context.intent_label_ids)

    context.domain_labels=list(context.df.domain.unique())
    context.domain_label_ids={}
    for l in context.df.domain.unique():
        context.domain_label_ids[l]=len(context.domain_label_ids)

    context.train_dataset=PandasDataset(args, context, "train")
    context.valid_dataset=PandasDataset(args, context, "val")
    context.test_dataset=PandasDataset(args, context, "test")
    context.test_dataset_all=PandasDataset(args, context, "test", filter_domains=False)

    num_workers=0

    train_batch_size=args.adapter_train_batch_size if args.add_adapter else args.non_adapter_train_batch_size
    test_batch_size=args.adapter_test_batch_size if args.add_adapter else args.non_adapter_test_batch_size
    
    context.train_dataloader=DataLoader(context.train_dataset, batch_size=train_batch_size, shuffle=shuffle_train, num_workers=num_workers)
    context.valid_dataloader=DataLoader(context.valid_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    context.test_dataloader=DataLoader(context.test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return context

class PandasDataset(Dataset):

    def __init__(self, args, context, dataset, subsample_train_ten=False, filter_domains=True):

        self.max_seq_length=args.max_seq_length
        self.tokenizer=context.tokenizer
        self.args=args
        self.task=context.task
        self.subsample_train_ten=subsample_train_ten
        self.df=context.df
        self.context=context

        self.lock=threading.Lock()

        # filter for domains
        if context.domains is not None and filter_domains:
            self.df=self.df[self.df.domain.apply(lambda x : x in context.domains)]

        # filter for dataset
        self.df=self.df[self.df.dataset==dataset]
        
        # subsample
        if args.subsample>0:
            self.df=self.df[0:args.subsample]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        try:
            self.lock.acquire()
            row=self.df.iloc[idx]
        finally:
            self.lock.release()

        x = tokenize(self.args, row["text"], self.tokenizer)
        if len(x)==3:
            input_ids, attention_mask, token_type_ids=x
        elif len(x)==2:
            input_ids, attention_mask=x
            token_type_ids=None
        else:
            raise Exception()

        sample={
            "idx": idx,
            "df_id": row["id"],
            "text": row["text"],
            "intent": row["intent"],
            "domain": row["domain"],
            "intent_encoded": torch.tensor(self.context.intent_label_ids[row["intent"]]),
            "domain_encoded": torch.tensor(self.context.domain_label_ids[row["domain"]]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        sample["labels"]=sample[self.context.task + "_encoded"]

        if token_type_ids is not None:
            sample["token_type_ids"]=token_type_ids

        # delete data that we do not need because they confuse huggingfaces trainer
        for key in list(sample.keys()):
            if key not in ['input_ids', 'attention_mask', 'labels', 'token_type_ids']:
                del sample[key]

        return sample

# load the assignment of domains to dataset
# n_modules determines on how many modules to split the domains 
def load_domain_assignment(args):

    inputfile=os.path.join(args.dataset_path, args.dataset + "_domains.json")
    f=open(inputfile, "r")
    domains=json.load(f)
    f.close()

    random.shuffle(domains)

    domain_assignment=[[] for i in range(args.num_modules)]
    for i in range(len(domains)):
        domain_assignment[i%args.num_modules].append(domains[i])
    
    return domain_assignment
