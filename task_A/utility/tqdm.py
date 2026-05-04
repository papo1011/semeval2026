from tqdm import tqdm
from xgboost.callback import TrainingCallback

class TqdmCallback(TrainingCallback):
    def __init__(self, n_estimators):
        self.pbar = tqdm(total=n_estimators, desc="Training XGBoost", unit="tree")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False 

    def after_training(self, model):
        self.pbar.close()
        return model