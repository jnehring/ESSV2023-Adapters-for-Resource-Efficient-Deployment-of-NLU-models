import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# read original data and create a dataframe out of it
df=pd.read_csv("results_duration.csv")
data=[]
for ix, row in df.iterrows():

    p=row["key"].split(",")
    p=[x.split("=") for x in p]
    p=[x[-1] for x in p]
    p.append(row["value"])
    data.append(p)

df=pd.DataFrame(data, columns=["model", "i", "dataset", "metric", "value"])
df=df[(df.metric=="train_duration") | (df.metric=="predict_duration")]

model_names={
    "bert+adapter": "BERT+Adapter",
    "bert-base-uncased": "BERT",
    "distilbert+adapter": "Distilbert+Adapter",
    "distilbert-base-uncased": "Distilbert",
    "svm": "SVM"
}
df.model=df.model.apply(lambda x : model_names[x])

# add number of samples in each set
num_samples={}
for dataset in df.dataset.unique():
    dfd=pd.read_csv("../data/" + dataset + ".csv")
    num_samples[dataset]={
        "train": len(dfd[dfd.dataset=="train"]),
        "test": len(dfd[dfd.dataset=="test"])
    }

num=[]
for ix, row in df.iterrows():
    n=num_samples[row["dataset"]]
    if row["metric"] == "train_duration":
        n=n["train"]
    else:
        n=n["test"]
    num.append(n)
df["num_samples"]=num

# convert to samples per second
df["value"]=df.num_samples/df.value

datasets = {
    "hwu": "HWU64-DG",
    "clinc": "CLINC150",
    "hwu_orig": "HWU64",
    "atis": "ATIS",
    "banking77": "Banking77",
    "": ""
}

dfp = df.copy()

figure(figsize=(3,3), dpi=80)
dfp = dfp.rename(columns={"model": "Model", "dataset": "Dataset", "metric": "Metric", "value": "Samples per second"})
dfp.Dataset = dfp.Dataset.apply(lambda x : datasets[x])
metrics = {"train_duration": "train", "predict_duration": "predict"}
dfp.Metric = dfp.Metric.apply(lambda x : metrics[x])
ax = sns.barplot(data=dfp[dfp.Metric=="predict"], x="Model", y="Samples per second")
plt.title("Prediction speed")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()
#plt.yscale('log')
outfile = "output/duration_barplot.pdf"
plt.savefig(outfile)
print("wrote " + outfile)

print(dfp)
sys.exit(0)

# average over the runs
data=[]
for model in df.model.unique():
    for dataset in df.dataset.unique():
        subdf=df[(df.model==model) & (df.dataset==dataset)]
        train_duration_mean=subdf[subdf.metric=="train_duration"].value.mean()
        predict_duration_mean=subdf[subdf.metric=="predict_duration"].value.mean()
        train_duration_std=subdf[subdf.metric=="train_duration"].value.std()
        predict_duration_std=subdf[subdf.metric=="predict_duration"].value.std()

        data.append([model, dataset, train_duration_mean, predict_duration_mean, train_duration_std, predict_duration_std])

df=pd.DataFrame(data, columns=["model", "dataset", "train_sps_mean", "predict_sps_mean", "train_sps_std", "predict_sps_std"])

print(df.dataset.unique())

rasa_values = [
    ["DIET", "atis", 47.26, 1.98, 127.28, 0.4],
    ["DIET", "banking77", 33.68, 3.57, 125.47, 1.37],
    ["DIET", "clinc", 53.85, 1.58, 131.88, 3.04],
    ["DIET", "hwu", 53.84, 1.41, 134.95, 1.77],
    ["DIET", "hwu_orig", 53.82, 1.72, 130.75, 0.75],
]

rasa_values = pd.DataFrame(rasa_values, columns = ['model', 'dataset', 'train_sps_mean', 'train_sps_std', 'predict_sps_mean',
       'predict_sps_std'])

df = pd.concat((df, rasa_values))


df["train_sps"]=df["train_sps_mean"].apply(lambda x : f"{x:.2f}") + " (" + df["train_sps_std"].apply(lambda x : f"{x:.2f}") + ")"
df["predict_sps"]=df["predict_sps_mean"].apply(lambda x : f"{x:.2f}") + " (" + df["predict_sps_std"].apply(lambda x : f"{x:.2f}") + ")"

means=[]


for model in df.model.unique():

    train_sps = df[df.model==model]["train_sps_mean"].mean()
    train_std = df[df.model==model]["train_sps_mean"].std()
    predict_sps = df[df.model==model]["predict_sps_mean"].mean()
    predict_std = df[df.model==model]["predict_sps_mean"].std()

    train_sps = f"{train_sps:.2f} ({train_std:.2f})"
    predict_sps = f"{predict_sps:.2f} ({predict_std:.2f})"

    row=[model, train_sps, predict_sps]

    means.append(row)

means=pd.DataFrame(means, columns=["model", "train_sps", "predict_sps"])
means = means.sort_values(by=["model"])

dataset_means=[]
for dataset in df.dataset.unique():
    train_duration = df[df.dataset == dataset].train_sps_mean.mean()
    train_std = df[df.dataset == dataset].train_sps_mean.std()
    test_duration = df[df.dataset == dataset].predict_sps_mean.mean()
    test_std = df[df.dataset == dataset].predict_sps_mean.std()

    train_duration = f"{train_duration:.2f} ({train_std:.2f})"
    test_duration = f"{test_duration:.2f} ({test_std:.2f})"

    dataset_means.append([datasets[dataset], train_duration, test_duration])

dataset_means=pd.DataFrame(dataset_means, columns=["dataset", "train_sps", "predict_sps"]).sort_values(by=["dataset"])
dataset_means = dataset_means.sort_values(by=["dataset"])
columns=["model", "dataset", "train_sps", "predict_sps"]

df=df[columns]
df=df.sort_values(["dataset", "model"])

df.dataset = df.dataset.apply(lambda x: datasets[x])

print("-"*20)
print("single results")
print("-"*20)
print()
print(df.to_latex(index=False))

print()
print("-"*20)
print("average model values")
print("-"*20)
print()
print(means.to_latex(index=False))

print()
print("-"*20)
print("average dataset values")
print("-"*20)
print() 
print(dataset_means.to_latex(index=False))
