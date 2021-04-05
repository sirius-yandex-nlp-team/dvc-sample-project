import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from dvc_sample_project.context import ctx
from dvc_sample_project.config import params
from dvc_sample_project.logger import logger, init_logger

import minikts.api as kts

def train_catboost(train_df, test_df):
    x_train = train_df.drop(["y"], axis=1)
    x_test = test_df.drop(["y"], axis=1)
    y_train = train_df["y"]
    y_test = test_df["y"]
    with kts.parse_stdout(kts.patterns.catboost, kts.LoggerCallback(logger=logger)):
        model = CatBoostClassifier(n_estimators=params.catboost.n_estimators)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])
    logger.log_metric("train_accuracy", accuracy_score(y_train, model.predict(x_train)), dvc=True)
    logger.log_metric("test_accuracy", accuracy_score(y_test, model.predict(x_test)), dvc=True)
    logger.dvclive_next_step()
    return model

if __name__ == "__main__":
    init_logger(tags=["debug"])
    train_df = pd.read_csv(ctx.data_dir / "processed" / "train.csv")
    test_df = pd.read_csv(ctx.data_dir / "processed" / "test.csv")
    catboost = train_catboost(train_df, test_df)
    pickle.dump(catboost, open(ctx.root_dir / "artifacts" / "catboost.pkl", "wb"))
