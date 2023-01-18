# determine sizes of models bert, bert+adapter, distilbert, distilbert+adapter

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, AutoModelWithHeads
import os
import shutil

num_labels=64

folder="/tmp/models/"

if os.path.exists(folder):
    shutil.rmtree(folder)
os.mkdir(folder)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_size(model, suffix):
    name=type(model).__name__.lower()
    model_folder=os.path.join(folder, name + suffix)

    os.mkdir(model_folder)

    model_file=os.path.join(model_folder, "model.bin")
    torch.save(model.state_dict(), model_file)
    folder_size=os.path.getsize(model_file)
    folder_size*=pow(10,-6)

    number_of_parameters=get_n_params(model)
    return folder_size, number_of_parameters

for model_name in ["distilbert-base-uncased", "bert-base-uncased"]:
    # load vanilla model
    model = AutoModelWithHeads.from_pretrained(model_name, num_labels=num_labels)
    size_without_adapter, n_parameters_without_adapter = get_size(model, "_pre")

    print()
    print("-"*20)
    print(f"model {model_name}")
    print(f"size without adapter {size_without_adapter} MB, {n_parameters_without_adapter:,} parameters.")

    # add adapter
    labels=[str(x) for x in range(num_labels)]
    label_to_id={}
    for label in labels:
        label_to_id[label]=len(label_to_id)

    model.add_adapter("modular_chatbot")
    model.add_classification_head(
        "modular_chatbot",
        num_labels=num_labels,
        id2label=label_to_id
    )
    model.train_adapter("modular_chatbot")
    size_with_adapter, n_parameters_with_adapter = get_size(model, "_post")

    print(f"size with adapter {size_with_adapter} MB, {n_parameters_with_adapter:,} parameters.")
    size_adapter=size_with_adapter-size_without_adapter
    print(f"size of adapter {size_adapter} MB")
    print("-"*20)
