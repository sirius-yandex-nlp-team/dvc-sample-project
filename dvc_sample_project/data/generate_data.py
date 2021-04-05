import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from dvc_sample_project.context import ctx
from dvc_sample_project.config import params

def generate_data():
    X, y = make_classification(params.data.n_objects)
    result = pd.DataFrame(X)
    result.columns = [f"feature_{i}" for i in range(X.shape[1])]
    result["y"] = y
    return result

if __name__ == "__main__":
    result = generate_data()
    result.to_csv(ctx.data_dir / "raw" / "data.csv", index=False)
