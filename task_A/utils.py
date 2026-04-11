from datasets import load_dataset
from tiktoken import get_encoding
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def load_dataset_XGB(config_name):
    dataset = load_dataset("DaniilOr/SemEval-2026-Task13", config_name)

    df_train = dataset["train"].to_pandas().reset_index(drop=True)
    df_val = dataset["validation"].to_pandas().reset_index(drop=True)
    df_test = dataset["test"].to_pandas().reset_index(drop=True)

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