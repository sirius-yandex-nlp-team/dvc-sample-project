import sys
import os
import pickle
import json

import sklearn.metrics as metrics

from dvc_sample_project.context import ctx
from dvc_sample_project.config import params
from dvc_sample_project.logger import logger, init_logger

import minikts.api as kts



def eval_classifier(model, matrix):
    targets = matrix[:, 1].toarray()
    X = matrix[:, 2:]

    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]
    
    precision, recall, prc_thresholds = metrics.precision_recall_curve(targets, predictions)
    prc_data = {'prc': [{
                'precision': p,
                'recall': r,
                'threshold': t} for p, r, t in zip(precision, recall, prc_thresholds)]
                }

    fpr, tpr, roc_thresholds = metrics.roc_curve(targets, predictions)
    roc_data = {'roc': [{
                'fpr': fp,
                'tpr': tp,
                'threshold': t} for fp, tp, t in zip(fpr, tpr, roc_thresholds)]
                }

    avg_prec = metrics.average_precision_score(targets, predictions)
    roc_auc = metrics.roc_auc_score(targets, predictions)
    AUCs = {'avg_prec': avg_prec, 'roc_auc': roc_auc}


    return AUCs, prc_data, roc_data



if __name__ == "__main__":

    init_logger(tags=['debug'])

    if len(sys.argv) != 6:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate.py model features scores prc roc\n')
        sys.exit(1)

    model_file = sys.argv[1]
    matrix_file = os.path.join(sys.argv[2], 'test.pkl')
    scores_file = sys.argv[3]
    prc_file = sys.argv[4]
    roc_file = sys.argv[5]

    with open(model_file, 'rb') as fd:
        model = pickle.load(fd)

    with open(matrix_file, 'rb') as fd:
        matrix = pickle.load(fd)

    
    AUCs, prc_data, roc_data = eval_classifier(model, matrix)
    
    with open(scores_file, 'w') as fd:
        json.dump({'avg_prec': avg_prec, 'roc_auc': roc_auc}, fd, indent=4)

    with open(prc_file, 'w') as fd:
        json.dump(prc_data, fd, indent=4)

    with open(roc_file, 'w') as fd:
        json.dump(roc_data, fd, indent=4)
