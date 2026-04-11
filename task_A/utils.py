from datasets import load_dataset
import pandas as pd

def load_dataset_XGB(config_name):
    dataset = load_dataset("DaniilOr/SemEval-2026-Task13", config_name)

    # load_dataset returns a DatasetDict for this benchmark
    df_train = dataset["train"].to_pandas().reset_index(drop=True)
    df_val = dataset["validation"].to_pandas().reset_index(drop=True)
    df_test = dataset["test"].to_pandas().reset_index(drop=True)

    # Print the sizes of the datasets
    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    #return the datasets as pandas DataFrames
    return df_train, df_val, df_test