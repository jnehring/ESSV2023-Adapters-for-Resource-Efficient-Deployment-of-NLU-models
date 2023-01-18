import argparse
import pandas as pd

def main():

    # parse arguments    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="results_grid_search.csv")
    parser.add_argument("--outdir", type=str, default="output")
    args=parser.parse_args()

    df=pd.read_csv(args.infile)

    data=[]
    for ix, row in df.iterrows():
        new_row=[]
        parts=row["key"].split(",")
        for part in parts:
            if part.find("=")>0:
                key,value=part.split("=")
                new_row.append(value)
            else:
                new_row.append(part)
        new_row.append(row["value"])
        data.append(new_row)
    df=pd.DataFrame(data, columns=["model", "learning_rate", "batch_size", "metric", "accuracy"])
    df=df[df.metric.apply(lambda x : x in ["acc_id_all"])]
    df.accuracy=df.accuracy.astype(float)

    for model in df.model.unique():
        model_df=df[df.model==model]
        best=model_df[model_df.accuracy==model_df.accuracy.max()].iloc[0]
        print(f"best performing parameter combination for model {model}")
        for key in ["learning_rate", "batch_size", "accuracy"]:
            val=best[key]
            if key=="accuracy":
                val=f"{100*val:.2f}%"
            print(key, val)
        print()
if __name__ == '__main__':
    main()

