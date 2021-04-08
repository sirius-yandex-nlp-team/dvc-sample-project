import sys
import os
import pickle
import json

from sklearn.metrics import mean_squared_error as MSE

from dvc_sample_project.context import ctx
from dvc_sample_project.config import params
from dvc_sample_project.logger import logger, init_logger

import minikts.api as kts


def eval_regressor(model, matrix):
    targets = matrix[:, 1].toarray()
    X = matrix[:, 2:]

    predictions = model.predict(X)
    rmse = MSE(y, predictions, squared=False)
    logger.log_metric("test RMSE", rmse, dvc=True)
    logger.dvclive_next_step()

    return rmse



if __name__ == "__main__":

    init_logger(tags=['debug'])

    if len(sys.argv) != 4:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate.py model-file features-dir scores-file\n')
        sys.exit(1)

    model_file = sys.argv[1]
    matrix_file = os.path.join(sys.argv[2], 'test.pkl')
    scores_file = sys.argv[3]

    with open(model_file, 'rb') as fd:
        model = pickle.load(fd)

    with open(matrix_file, 'rb') as fd:
        matrix = pickle.load(fd)
    
    rmse = eval_regressor(model, matrix)
    
    with open(scores_file, 'w') as fd:
        json.dump({'RMSE': rmse}, fd, indent=4)