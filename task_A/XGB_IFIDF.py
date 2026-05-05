import utility.utils as utils
from utility.tqdm import TqdmCallback
from xgboost.callback import EarlyStopping
from xgboost import XGBClassifier
import numpy as np
#import cudf

def best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        scores.append(utils.f1_score(y_true, y_pred))
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx]

def train_eval(X_train, X_test,X_val, y_train, y_test, y_val):
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        device="cpu",
        eval_metric="auc",
        callbacks=[EarlyStopping(rounds=100)]
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
     #  Probabilità su validation (per threshold tuning)
    y_val_prob = xgb.predict_proba(X_val)[:, 1]
    thr, best_f1 = best_threshold(y_val, y_val_prob)
    print(f"Best threshold (val): {thr:.3f}")
    print(f"Best validation F1: {best_f1:.4f}")
    
     #  Debug distribuzione (super utile)
    print("Val prob mean:", y_val_prob.mean())

    #Probabilità su test    
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= thr).astype(int)
    utils.save_results(
        y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="XGB_TFIDF"
    )
def run_on_problems(df_train, df_test, df_val):
    y_train = df_train["label"].values
    y_test = df_test["label"].values
    y_val = df_val["label"].values
    print("Y train size:", y_train.shape)
    print("Y test size:", y_test.shape)
    print("Y val size:", y_val.shape)
    # 1. Train transform
    X_train, embedder = utils._tfidf(corpus=df_train["code"].values, max_features=8000)
    

    X_test = embedder.transform(df_test["code"].values)
    X_val = embedder.transform(df_val["code"].values)

    print("X train size:", X_train.shape)
    print("X test size:", X_test.shape)
    print("X val size:", X_val.shape)

    train_eval(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, X_val=X_val, y_val=y_val)

def run():
    df_train, df_val, df_test = utils.load_dataset_XGB("A")
    run_on_problems(df_train=df_train, df_test=df_test, df_val=df_val)
