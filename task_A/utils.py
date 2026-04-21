from datasets import load_dataset
from tiktoken import get_encoding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import numpy as np
import os

def load_dataset_XGB(config_name):
    print("Loading Dataset")
    dataset = load_dataset("DaniilOr/SemEval-2026-Task13", config_name)

    df_train = dataset["train"].to_pandas().reset_index(drop=True)
    df_val = dataset["validation"].to_pandas().reset_index(drop=True)
    df_test = dataset["test"].to_pandas().reset_index(drop=True)

    print(df_train.shape)

    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    return df_train, df_val, df_test

def _tokenize(corpus):
    tokenizer = get_encoding("cl100k_base")
    tokenized_corpus = [tokenizer.encode(text) for text in corpus]
    return tokenized_corpus

def _tfidf(corpus, max_features=10000):
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, max_features=max_features)
    return vectorizer.fit_transform(_tokenize(corpus)).toarray(), vectorizer

def save_results(y_test, y_pred, y_prob, file_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(y_test, y_prob)
    if os.path.exists(f"Results/{file_name}.npz"):
        results = np.load(f"Results/{file_name}.npz")
        accuracy = np.append(results["accuracy"], accuracy)
        precision = np.append(results["precision"], precision)
        recall = np.append(results["recall"], recall)
        f1 = np.append(results["f1"], f1)
        auc = np.append(results["auc"], auc)
    np.savez(
        f"Results/{file_name}.npz",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
    )