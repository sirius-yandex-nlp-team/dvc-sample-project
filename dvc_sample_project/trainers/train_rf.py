import sys
import os
import pickle
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

from dvc_sample_project.context import ctx
from dvc_sample_project.config import params
from dvc_sample_project.logger import logger, init_logger

import minikts.api as kts

def train_random_forest(train_matrix):

	seed = params.seed
	n_est = params.n_est
	min_split = params.min_split

	y = np.squeeze(matrix[:, 1].toarray())
	X = matrix[:, 2:]

	with kts.parse_stdout(kts.patterns.random_forest, kts.LoggerCallback(logger=logger)):
		model = RandomForestRegressor(
		    n_estimators=n_est,
		    min_samples_split=min_split,
		    n_jobs=2,
		    random_state=seed
		)
		model.fit(X, y)

	logger.log_metric("RMSE", MSE(y, model.predict(X), squared=False), dvc=True)
	logger.dvclive_next_step()

	return model



if __name__ == "__main__":
	if len(sys.argv) != 3:
	    sys.stderr.write('Arguments error. Usage:\n')
	    sys.stderr.write('\tpython train.py features-dir model-dir\n')
	    sys.exit(1)

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	init_logger(tags=["debug"])

	with open(os.path.join(input_path, 'train.pkl'), 'rb') as fd:
	    matrix = pickle.load(fd)

	rf_clf = train_random_forest(matrix)

	with open(os.path.join(output_path, 'RF_model.pkl'), 'wb') as fd:
	    pickle.dump(rf_clf, fd)
