import utility.utils as utils
from utility.tqdm import TqdmCallback
from xgboost import XGBClassifier

def train_eval(X_train, X_test, y_train, y_test):
    n_trees = 100

    xgb = XGBClassifier(n_estimators=n_trees, 
                        callbacks=[TqdmCallback(n_estimators=n_trees)])
    
    xgb.fit(X_train, 
            y_train)
    
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    utils.save_results(
        y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="XGB_TFIDF"
    )

def run_on_problems(df_train, df_test):

    y_train = df_train["label"].values
    y_test = df_test["label"].values

    print("Y train size:", y_train.shape)
    print("Y test size:", y_test.shape)

    X_train, embedder = utils._tfidf(corpus=df_train["code"].values, max_features=1536)
    X_test = embedder.transform(utils._tokenize(corpus=df_test["code"].values)).toarray()

    print("X train size:", X_train.shape)
    print("X test size:", X_test.shape)


    train_eval(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run():
    df_train, _, df_test = utils.load_dataset_XGB("A")
    run_on_problems(df_train=df_train, df_test=df_test)