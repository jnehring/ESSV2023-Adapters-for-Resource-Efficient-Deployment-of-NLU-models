import pandas as pd
import numpy as np

data = []
for dataset in ["atis", "banking77", "clinc", "hwu", "hwu_orig"]:
    df = pd.read_csv(f"data/{dataset}.csv")
    mean_utterance_length = np.mean(df.text.apply(lambda x: len(x)))
    std_utterance_length = np.std(df.text.apply(lambda x: len(x)))
    max_utterance_length = np.max(df.text.apply(lambda x: len(x)))

    mean_utterance_length = f"{mean_utterance_length:.2f} ({std_utterance_length:.2f})"  

    num_samples = len(df)
    num_intents = len(set(df.intent))
    num_domains = len(set(df.domain))

    row = [dataset, num_samples, num_domains, num_intents, max_utterance_length, mean_utterance_length]
    data.append(row)

df=pd.DataFrame(data, columns=["Dataset", "# Samples", "# Domains", "# Intents", "Max Utterance Length", "Mean Utterance Length"])

datasets = {
    "hwu": "HWU64-DG",
    "clinc": "CLINC150",
    "hwu_orig": "HWU64",
    "atis": "ATIS",
    "banking77": "Banking77",
    "": ""
}
df.Dataset = df.Dataset.apply(lambda x: datasets[x])

for c in ["# Samples", "# Domains", "# Intents"]:
    df[c] = df[c].apply(lambda x : f"{x:,}")

# df["Dataset Length"] = df["Dataset Length"].apply(lambda x : f"{x}")
print(df.to_latex(index=False))