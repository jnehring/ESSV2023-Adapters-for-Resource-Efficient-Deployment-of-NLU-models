import os
import urllib.request
import pandas as pd
import json


# download banking77 dataset and parse it to our data format

def download_data():
    tmp_dir = "/tmp/"
    url = "https://github.com/howl-anderson/ATIS_dataset/raw/master/data/standard_format/rasa/"
    files = []
    for file in ["train.json", "test.json"]:
        remote_file = url + file
        local_file = os.path.join(tmp_dir, file)

        if os.path.exists(local_file):
            print("use " + file + " from cache")
        else:
            urllib.request.urlretrieve(remote_file, local_file)
            print("downloading " + file)
        files.append(local_file)
    return files[0], files[1]


def parse_data(infile, dataset):
    data = json.load(open(infile, "r"))
    df = []
    for sample in data["rasa_nlu_data"]["common_examples"]:
        df.append([
            sample["text"],
            sample["intent"],
            "atis",
            dataset
        ])
    df = pd.DataFrame(df, columns=["text", "intent", "domain", "dataset"])
    return df


if __name__ == "__main__":
    train_file, test_file = download_data()

    df_train = parse_data(train_file, "train")
    df_valid = df_train.sample(frac=0.1)
    df_train = df_train.drop(df_valid.index)
    df_valid.dataset = "val"
    df_train.dataset = "train"
    df_test = parse_data(test_file, "test")

    df = pd.concat([df_train, df_valid, df_test])
    outfile = "data/atis.csv"
    df.to_csv(outfile)
    print("wrote " + outfile)

    domains_file = "data/atis_domains.json"
    domains = ["atis"]
    f = open(domains_file, "w")
    f.write(json.dumps(domains))
    f.close()
    print("wrote " + domains_file)
