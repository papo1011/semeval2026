from datasets import load_dataset
import pandas as pd

def load_dataset_XGB():
    dataset = load_dataset("DaniilOr/SemEval-2026-Task13")
    data = list(dataset)
    df = pd.DataFrame(data)
    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    df_val = df[df['split'] == 'validation'].reset_index(drop=True)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    return df_train, df_val, df_test