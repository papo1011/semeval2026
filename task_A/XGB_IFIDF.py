import utility.utils as utils
from utility.tqdm import TqdmCallback
from xgboost import XGBClassifier
import numpy as np
import cudf

def best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        scores.append(utils.f1_score(y_true, y_pred))
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx]

def train_eval(X_train, X_val, X_test, y_train, y_val, y_test):
    n_trees = 3000
    counter = np.bincount(y_train)
    ratio = counter[0] / counter[1]

    xgb = XGBClassifier(n_estimators=n_trees,
                        learning_rate=0.03,
                        max_depth=6,
                        min_child_weight=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=1e-3,
                        reg_lambda=1.0,
                        callbacks=[TqdmCallback(n_estimators=n_trees)],
                        tree_method="hist",
                        device="cuda",
                        eval_metric="auc"
                        )
    
    xgb.fit(X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
            early_stopping_rounds=100,)
    
     #  Probabilità su validation (per threshold tuning)
    y_val_prob = xgb.predict_proba(X_val)[:, 1]
    thr, best_f1 = best_threshold(y_val, y_val_prob)

    print(f"Best threshold (val): {thr:.3f}")
    print(f"Best validation F1: {best_f1:.4f}")

     #  Debug distribuzione (super utile)
    print("Val prob mean:", y_val_prob.mean())

    #  Probabilità su test
    y_test_prob = xgb.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= thr).astype(int)


    utils.save_results(
        y_test=y_test, y_pred=y_test_pred, y_prob=y_test_prob, file_name="XGB_TFIDF"
    )

    return y_val_prob, y_test_prob #utili per eventuale ensemble con altri modelli

def run_on_problems(df_train, df_test, df_val):

    y_train = df_train["label"].values
    y_test = df_test["label"].values
    y_val = df_val["label"].values


    # 1. Train transform
    X_train, embedder = utils._tfidf(corpus=df_train["code"].values, max_features=10000)
    
    gdf_val = cudf.Series(df_val["code"].values)
    X_val = embedder.transform(gdf_val).get()

    gdf_test = cudf.Series(df_test["code"].values)
    X_test = embedder.transform(gdf_test).get()

    print("Y train size:", y_train.shape)
    print("Y test size:", y_test.shape)
    print("Y val size:", y_val.shape)


    train_eval(X_train=X_train,X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)


def run():
    df_train, df_val, df_test = utils.load_dataset_XGB("A")
    val_prob, test_prob = run_on_problems(df_train=df_train, df_val=df_val, df_test=df_test)
    np.save("xgb_val_prob.npy", val_prob)
    np.save("xgb_test_prob.npy", test_prob)