import utility.utils as utils
from utility.tqdm import TqdmCallback
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

def train_eval(X_train, X_test, X_val, y_train, y_test, y_val):
    n_trees = 300
    counter = np.bincount(y_train)
    ratio = counter[0] / counter[1]
    xgb = XGBClassifier(n_estimators=n_trees,
                        learning_rate=0.07,
                        max_depth=5,
                        scale_pos_weight=ratio,
                        callbacks=[TqdmCallback(n_estimators=n_trees)],
                        tree_method="hist",
                        device="cuda")
    
    xgb.fit(X_train, 
            y_train)
    y_prob = xgb.predict_proba(X_val)[:, 1]
    thr = best_threshold(y_val, y_prob)

    y_pred = (y_prob >= thr).astype(int)
    
    utils.save_results(
            y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="XGB_TFIDF")
def run_on_problems(df_train, df_test):
    y_train = df_train["label"].values
    y_test = df_test["label"].values
    print("Y train size:", y_train.shape)
    print("Y test size:", y_test.shape)
    # 1. Train transform
    X_train, embedder = utils._tfidf(corpus=df_train["code"].values, max_features=4000)
    
    gdf_test = cudf.Series(df_test["code"].values)
    X_test = embedder.transform(gdf_test).get()
    print("X train size:", X_train.shape)
    print("X test size:", X_test.shape)

    train_eval(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

def run():
    df_train, _, df_test = utils.load_dataset_XGB("A")
    run_on_problems(df_train=df_train, df_test=df_test)

