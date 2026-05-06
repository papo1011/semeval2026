import utility.utils as utils
from utility.tqdm import TqdmCallback
from xgboost import EarlyStopping
from xgboost import XGBClassifier
import numpy as np
import cudf

def best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.2, 0.7, 50)
    best_thr = 0.5
    best_f1 = 0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = utils.f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr

def train_eval(X_train, X_test,X_val, y_train, y_test, y_val):
    n_trees = 1000
    counter = np.bincount(y_train)
    ratio = counter[0] / counter[1]
    xgb = XGBClassifier(n_estimators=n_trees,
                        learning_rate=0.05,
                        max_depth=6,
                        callbacks=[TqdmCallback(n_estimators=n_trees), EarlyStopping(rounds=100)],
                        tree_method="hist",
                        device="cuda")
    
    xgb.fit(X_train, 
            y_train,
            eval_set=[(X_val, y_val)])
    
    
    y_prob = xgb.predict_proba(X_test)[:, 1]
    best_threshold_val = best_threshold(y_val, xgb.predict_proba(X_val)[:, 1])
    y_pred = (y_prob >= best_threshold_val).astype(int)

    utils.save_results(
            y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="XGB_TFIDF")
    
def run_on_problems(df_train, df_test, df_val):
    y_train = df_train["label"].values
    y_test = df_test["label"].values
    y_val = df_val["label"].values
    print("Y train size:", y_train.shape)
    print("Y test size:", y_test.shape)
    print("Y val size:", y_val.shape)
    # 1. Train transform
    X_train, embedder = utils._tfidf(corpus=df_train["code"].values, max_features=2000)

    gdf_val = cudf.Series(df_val["code"].values)
    X_val = embedder.transform(gdf_val).get()
    
    gdf_test = cudf.Series(df_test["code"].values)
    X_test = embedder.transform(gdf_test).get()

    print("X train size:", X_train.shape)
    print("X test size:", X_test.shape)

    train_eval(X_train=X_train, X_test=X_test, X_val=X_val, y_train=y_train, y_test=y_test, y_val=y_val)

def run():
    df_train, df_val, df_test = utils.load_dataset_XGB("A")
    run_on_problems(df_train=df_train, df_test=df_test, df_val=df_val)
