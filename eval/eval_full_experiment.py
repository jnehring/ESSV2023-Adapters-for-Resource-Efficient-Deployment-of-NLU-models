# todo

# put plots in the paper, if possible as .pdf file. somehow there was a problem and I could not add graphics to the paper
# add results of mehri to the results table accuracy_nonmod.tex.
# research size of models in parameter
# research size of rasa model in MB

import argparse
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

# parse result file
def read_data(args):
    df=pd.read_csv(os.path.join(args.infile))
    data=[]
    for ix, row in df.iterrows():
        new_row=[]
        parts=row["key"].split(",")
        for part in parts[0:4]:
            key,value=part.split("=")
            new_row.append(value)
        new_row.append(parts[4])
        new_row.append(row["value"])
        data.append(new_row)
    df=pd.DataFrame(data, columns=["dataset", "add_adapter", "num_modules", "model", "metric", "value"])
    df=df[df.metric.apply(lambda x : x in ["acc_id_all", "acc_ms", "train_duration", "predict_duration", "train_num_samples", "predict_num_samples"])]
    df.num_modules=df.num_modules.astype(int)
    df.value=df.value.astype(float)
    df.add_adapter=df.add_adapter.apply(lambda x : x == "True")

    datasets={
        "hwu": "HWU64-DG",
        "clinc": "CLINC150",
        "hwu_orig": "HWU64",
        "atis": "ATIS",
        "banking77": "Banking77"
    }
    df.dataset=df.dataset.apply(lambda x : datasets[x])
    model_names={
        "bert+adapter": "BERT+Adapter",
        "bert-base-uncased": "BERT",
        "distilbert+adapter": "Distilbert+Adapter",
        "distilbert-base-uncased": "Distilbert",
        "svm": "SVM"
    }
    df.model=df.model.apply(lambda x : model_names[x])

    return df

def get_result(df, filter, column):
    for key, value in filter.items():
        df=df[df[key]==value]
    assert(len(df)==1)
    return df.iloc[0][column]

def plot_accuracy_modular(args, results):

    print("="*30)
    print("modular accuracy")
    print("="*30)

    dfs=[]
    _results=results[results.metric=="acc_id_all"].copy()
    #_results["model"] = _results["add_adapter"].apply(lambda x : "Bert+Adapter" if x else "Bert")

    models=sorted(_results.model.unique())
    datasets=sorted(_results.dataset.unique())
    for num_modules in _results.num_modules.unique():
        df=[]
        for model in models:
            row=[model, num_modules]
            for dataset in datasets:

                if dataset in ("ATIS", "Banking77") and num_modules > 1:
                    acc="-"
                else:
                    fltr={
                        "num_modules": num_modules,
                        "dataset": dataset,
                        "model": model
                    }
                    acc=get_result(_results, fltr, "value")
                row.append(acc)
            df.append(row)

        columns=["model", "n modules"]
        columns.extend(datasets)
        df=pd.DataFrame(df, columns=columns)

        dfs.append(df)

    results = [
        {
            "model": "BERT (mehri20)",
            "n_modules": 1,
            "Banking77": 93.02,
            "CLINC150": 95.93,
            "HWU64-DG": 93.87,
        },
        {
            "model": "ConvBERT (mehri20)",
            "n_modules": 1,
            "Banking77": 92.95,
            "CLINC150": 97.07,
            "HWU64-DG": 90.43,
        },   
        {
            "model": "ConvBERT+ (mehri21)",
            "n_modules": 1,
            "Banking77": 93.83,
            "CLINC150": 97.31,
            "HWU64-DG": 93.03,
        },
        {
            "model": "DIET",
            "n_modules": 1,
            "ATIS": 95.29,
            "Banking77": 88.54,
            "CLINC150": 87.97,
            "HWU64": 84.48,
            "HWU64-DG": 87.98,
        },
        {
            "model": "DIET",
            "n_modules": 3,
            "CLINC150":  87.25,
            "HWU64": 85.05,
            "HWU64-DG": 88.80,
        },
        {
            "model": "DIET",
            "n_modules": 10,
            "CLINC150": 87.65,
            "HWU64": 83.55,
            "HWU64-DG": 89.28,
        },
    ]


    data = []
    for result in results:
        row = [result["model"], result["n_modules"]]
        for c in datasets:
            if c in result.keys():
                row.append(result[c]/100)
            else:
                row.append("-")
        data.append(row)
    dfs.append(pd.DataFrame(data, columns=columns))

    dfs=pd.concat(dfs).sort_values(by=["n modules", "model"])

    full_data=dfs.copy()

    models = full_data[full_data["n modules"]==3].model.unique()
    full_data = full_data[full_data.model.apply(lambda x : x in models)]
    data = []

    for ix, row in full_data.iterrows():
        avg=[]
        for c in ['CLINC150', 'HWU64', 'HWU64-DG']:
            avg.append(100*row[c])
        data.append([row["model"], str(row["n modules"]), np.mean(avg)])

    columns = ["Model", "Number of Modules", "Accuracy [%]"]
    data = pd.DataFrame(data, columns=columns)

    avg_data = []
    for n in data["Number of Modules"].unique():
        avg = np.mean(data[data["Number of Modules"] == n]["Accuracy [%]"])
        avg_data.append(["Mean", n, avg])
    avg_data = pd.DataFrame(avg_data, columns=columns)
    data = pd.concat((data, avg_data))

    # plt.clf()
    # plt.figure(figsize = (8,4))
    # ax = plt.subplot(111)
    # sns.lineplot(ax=ax, data=data, x="Number of Modules", y="Accuracy [%]", hue="Model")
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.savefig("output/modular_accuracy_plot.pdf")
    # return

    averages = []
    for ix, row in dfs.iterrows():
        avg=[]
        for c in datasets:
            if type(row[c]) == float:
                avg.append(100*row[c])
        m=np.mean(avg)
        std=np.std(avg)
        line = f"{m:.2f}% ({std:.2f})"
        averages.append(line)
        
    dfs["Mean"] = averages

    # dfs=dfs.reindex(columns=['n modules', 'model', 'ATIS', 'Banking77', 'CLINC150', 'HWU64','HWU64-DG', 'Mean'])
    dfs = dfs[dfs["n modules"]==1]
    dfs = dfs[dfs.model.apply(lambda x : x.find("mehri")<0)]

    df_plot = dfs.copy()
    def format(x):
        if type(x)==str:
            return x
        else:
            return f"{100*x:.2f}%"
    for c in datasets:
        dfs[c]=dfs[c].apply(format)

    dfs=dfs.reindex(columns=['model', 'ATIS', 'Banking77', 'CLINC150', 'HWU64', 'Mean'])
    dfs=dfs.rename(columns={"model": "Model"})

    print(dfs.to_latex(index=False))
    print()

    # plot the data for the poster
    df_plot=df_plot.reindex(columns=['model', 'ATIS', 'Banking77', 'CLINC150', 'HWU64', 'Mean'])
    df_plot=df_plot.rename(columns={"model": "Model"})

    new_df = {
        "Model": [],
        "F1-Score": [],
        "Dataset": []
    }
    for column in ['ATIS', 'Banking77', 'CLINC150', 'HWU64']:
        new_df["Model"].extend(df_plot.Model)
        new_df["F1-Score"].extend(df_plot[column])
        new_df["Dataset"].extend([column] * len(df_plot))

    new_df = pd.DataFrame(new_df)
    
    figure(figsize=(3,5), dpi=80)
    ax = sns.barplot(data=new_df, x="Dataset", y="F1-Score", hue="Model")
    plt.title("F1-score of intent detection")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.ylim([0.7, 1.0])
    plt.legend(bbox_to_anchor =(0.1, 1.25), title="Model")

    plt.tight_layout()
    outfile = "output/accuracy_barplot.pdf"
    plt.savefig(outfile)
    print("wrote " + outfile)

def plot_runtime(args, results):

    print("="*30)
    print("runtime")
    print("="*30)

    _results=results.copy()
    _results["model"] = results["add_adapter"].apply(lambda x : "Bert+Adapter" if x else "Bert")

    results_train=_results[(_results.metric=="train_duration") & (_results.num_modules==1)]
    results_predict=_results[(_results.metric=="predict_duration") & (_results.num_modules==1)]

    df={"model": results_train.model, "dataset": results_train.dataset, "train_duration": list(results_train.value), "predict_duration": list(results_predict.value)}
    df=pd.DataFrame(df)

    df_bert=df[df.model=="Bert"]
    df_adapter=df[df.model=="Bert+Adapter"]

    df={
        "dataset": df_bert.dataset,
        "train_bert": list(df_bert.train_duration),
        "train_adapter": list(df_adapter.train_duration),
        "predict_bert": list(df_bert.predict_duration),
        "predict_adapter": list(df_adapter.predict_duration)
    }
    df=pd.DataFrame(df)
    df["delta_train"]=100*df.train_adapter/df.train_bert
    df["delta_predict"]=100*df.predict_adapter/df.predict_bert
    
    def get_num_samples(infile, subset):
        df=pd.read_csv(infile)
        l=len(df[df.dataset==subset])
        return l
        #return f"{round(l/1000,1)}k"

    df["num_samples_train"]=[
        get_num_samples("../data/hwu.csv", "train"),
        get_num_samples("../data/clinc.csv", "train"),
        get_num_samples("../data/hwu_orig.csv", "train")
    ]

    df["num_samples_test"]=[
        get_num_samples("../data/hwu.csv", "test"),
        get_num_samples("../data/clinc.csv", "test"),
        get_num_samples("../data/hwu_orig.csv", "test")
    ]

    df.train_bert = df.num_samples_train / df.train_bert
    df.train_adapter = df.num_samples_train / df.train_adapter
    df.predict_bert = df.num_samples_test / df.predict_bert
    df.predict_adapter = df.num_samples_test / df.predict_adapter

    for c in ["train_bert", "train_adapter", "predict_bert","predict_adapter", "delta_train", "delta_predict"]:
        df[c]=df[c].apply(lambda x : f"{x:.2f}")

    df.delta_train=df.delta_train.apply(lambda x : f"{x}%")
    df.delta_predict=df.delta_predict.apply(lambda x : f"{x}%")
    df.num_samples_train=df.num_samples_train.apply(lambda l : f"{l:,}")


    order=["dataset", "num_samples_train", "train_bert", "train_adapter", "delta_train", "num_samples_test", "predict_bert","predict_adapter", "delta_predict"]
    df=df[order]

    print(df)

    #results["model"]=results["add_adapter"].apply(lambda x : "BERT+Adapter" if x else "Bert")
    #results.value=results.value/60


def plot_accuracy_nonmod(args, results):

    print("="*30)
    print("non modular accuracy")
    print("="*30)

    fltr={
        "num_modules": 1,
        "metric": "acc_id_all",
        "dataset": "HWU64",
        "add_adapter": True
    }
    datasets=["HWU64", "HWU64-DG", "CLINC150"]
    models=["Bert", "Bert+Adapter"]

    data=[]
    for model in models:
        fltr["add_adapter"]=model=="Bert+Adapter"
        row=[model]
        for dataset in datasets:
            fltr["dataset"]=dataset
            row.append(get_result(results, fltr, "value"))
        data.append(row)

    columns=["model"]
    columns.extend(datasets)
    data=pd.DataFrame(data, columns=["Model", "HWU64", "HWU64-DG", "CLINC150"])
    
    for dataset in datasets:
        data[dataset]=data[dataset].apply(lambda x : f"{100*x:.2f}%")
    
    outfile=os.path.join(args.outdir, "accuracy_nonmod.tex")
    f=open(outfile, "w")
    f.write(data.to_latex(index=False))
    f.close()

    print(data)

    print("produced " + outfile)
    print()

def model_size_table(args):
    # from get_model_sizes.py

    # model distilbert-base-uncased
    # size without adapter 265.491897 MB, 66,362,880 parameters.
    # size with adapter 269.854273 MB, 67,449,952 parameters.
    # size of adapter 4.362375999999983 MB

    # model bert-base-uncased
    # size without adapter 438.013105 MB, 109,482,240 parameters.
    # size with adapter 444.174921 MB, 111,016,576 parameters.
    # size of adapter 6.1618159999999875 MB

    print("="*30)
    print("model size")
    print("="*30)

    # to determine these we trained models and serialized them to disk
    size_bert=438.013105
    size_adapter_bert=6.1618159999999875
    size_rasa=64509*pow(10,-3)
    size_distilbert=265.491897
    size_adapter_distilbert=4.362375999999983
    size_svm=7192313*pow(10,-6)
    table=[
        ["BERT", f"{size_bert:.2f}"],
        ["BERT+Adapter", f"{size_bert:.2f} + n times {size_adapter_bert:.2f}"],
        ["Distilbert", f"{size_distilbert:.2f}"],
        ["Distilbert+Adapter", f"{size_distilbert:.2f} + n times {size_adapter_distilbert:.2f}"],
        ["DIET", f"{size_rasa:.2f}"],
        ["SVM", f"{size_svm:.2f}"]
    ]
    table=pd.DataFrame(table, columns=["Model", "Size [MB]"])
    #table["Size [MB]"]=table["Size [MB]"].apply(lambda x : f"{x:.2f}")
    print(table.to_latex(index=False))

    x0=1
    x1=40

    columns=["Model", "n", "size"]
    df=[
        ["Bert", x0, (x0*size_bert)/1000],
        ["Bert", x1, (x1*size_bert)/1000],
        ["Bert+Adapter", x0, (size_bert + x0*size_adapter_bert)/1000],
        ["Bert+Adapter", x1, (size_bert + x1*size_adapter_bert)/1000],
        ["DIET", x0, (x0*size_rasa)/1000],
        ["DIET", x1, (x1*size_rasa)/1000],
        ["Distilbert", x0, (x0*size_distilbert)/1000],
        ["Distilbert", x1, (x1*size_distilbert)/1000],
        ["Distilbert+Adapter", x0, (size_distilbert+x0*size_adapter_distilbert)/1000],
        ["Distilbert+Adapter", x1, (size_distilbert+x1*size_adapter_distilbert)/1000],
        ["SVM", x0, (x0*size_svm)/1000],
        ["SVM", x1, (x1*size_svm)/1000],
    ]
    df=pd.DataFrame(df, columns=columns)

    f_size = 25
    
    plt.clf()
    # sns.set(rc={'figure.figsize':(15,8)})
    plt.figure(figsize = (3, 2.5))
    ax = sns.lineplot(data=df, x="n", y="size", hue="Model")
    plt.title("Comparison of model size")
    ax.get_legend().remove()
    plt.xlabel("number of NLU models")
    plt.ylabel("size [GB]")
    # plt.legend(fontsize = f_size)
    #plt.xticks(fontsize=f_size)
    #plt.yticks(fontsize=f_size)
    # plt.xlim(1,40)
    plt.ylim(0,10)
    plt.tight_layout()

    outfile=os.path.join(args.outdir, "size_comparison.pdf")
    plt.savefig(outfile)
    print("produced " + outfile)
    print()

def main():

    # parse arguments    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="results_full_experiment.csv")
    parser.add_argument("--outdir", type=str, default="output")
    args=parser.parse_args()

    # make output dir
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # parse result file
    results=read_data(args)

    # non modular accuracy
    # plot_accuracy_nonmod(args, results)

    # model size table
    model_size_table(args)

    # accuracy in the modular scenario
    plot_accuracy_modular(args, results)

    # runtime
    # plot_runtime(args, results)

if __name__ == '__main__':
    main()

