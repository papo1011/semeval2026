from datasets import load_dataset
from tiktoken import get_encoding
import cudf
from cuml.feature_extraction.text import TfidfVectorizer as GPU_TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
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
    tokenized_corpus = [tokenizer.encode(text, disallowed_special=()) for text in corpus]
    return tokenized_corpus

def _tfidf(corpus, max_features=50000):

    gdf_corpus = cudf.Series(corpus)

    vectorizer = GPU_TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )

    X_tfidf_gpu = vectorizer.fit_transform(gdf_corpus)
    
    return X_tfidf_gpu.get(), vectorizer

def save_results(y_test, y_pred, y_prob, file_name, save_dir="/content/drive/MyDrive/"):
    """
    Salva i risultati nella Home di Google Drive. 
    Se il file esiste già, appende i nuovi risultati a quelli precedenti.
    """
    
    # 1. Calcolo delle metriche in modo pulito
    # Usiamo f1_score di sklearn per evitare errori di divisione per zero manuali
    current_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob)
    }
    
    # 2. Gestione del percorso
    # Creiamo la cartella se non esiste (anche se MyDrive esiste sempre dopo il mount)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            print(f"Errore: Impossibile accedere a {save_dir}. Hai montato il Drive?")
            return

    file_path = os.path.join(save_dir, f"{file_name}.npz")
    
    # 3. Caricamento dati esistenti e concatenazione
    if os.path.exists(file_path):
        try:
            existing_data = np.load(file_path)
            # Aggiorniamo ogni metrica nel dizionario aggiungendo il nuovo valore all'array esistente
            for key in current_metrics:
                if key in existing_data:
                    current_metrics[key] = np.append(existing_data[key], current_metrics[key])
                else:
                    # Se per qualche motivo manca una chiave, la inizializziamo come array
                    current_metrics[key] = np.array([current_metrics[key]])
        except Exception as e:
            print(f"Avviso: Errore nel caricamento del file esistente ({e}). Creazione nuovo file.")
    else:
        # Se il file è nuovo, trasformiamo i singoli valori in array per mantenere coerenza
        for key in current_metrics:
            current_metrics[key] = np.array([current_metrics[key]])

    # 4. Salvataggio definitivo
    np.savez(file_path, **current_metrics)
    print(f"Risultati aggiornati con successo in: {file_path}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))