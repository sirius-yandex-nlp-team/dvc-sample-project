import pandas as pd
from sklearn.model_selection import train_test_split

from dvc_sample_project.context import ctx
from dvc_sample_project.config import params

def split_data(df):
    train_df, test_df = train_test_split(
        df, 
        test_size=params.split.test_size,
        random_state=params.split.random_state,
    )
    return train_df, test_df

if __name__ == "__main__":
    df = pd.read_csv(ctx.data_dir / "raw" / "data.csv")
    train_df, test_df = split_data(df)
    train_df.to_csv(ctx.data_dir / "processed" / "train.csv", index=False)
    test_df.to_csv(ctx.data_dir / "processed" / "test.csv", index=False)
