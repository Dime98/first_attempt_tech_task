import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_pd_series(series, label_encoder):
    if not label_encoder:
        label_encoder = LabelEncoder()
    if series.dtype == object:
        return label_encoder.fit_transform(series)
    else:
        return series


def encode_pd_df(df, label_encoder=None):
    for column in df.columns:
        df[column] = encode_pd_series(df[column], label_encoder)
    return df


def handle_nulls(df):
    null_sum = df.isnull().sum()
    if np.any(null_sum.values > 0):
        null_columns_index = np.where(null_sum.values != 0)[0]
        null_columns = [df.columns[index] for index in null_columns_index]

        for column in null_columns:
            if df[column].dtype == object:
                df[column] = df[column].fillna("")
            elif df[column].dtype in [np.float64, np.int64]:
                df[column] = df[column].fillna(df[column].mean())
            else:
                raise NotImplementedError(f"Data type {df[column].dtype} not handled")
    return df


def drop_duplicates(df):
    if df.duplicated().sum() != 0:
        df.drop_duplicates(keep="first", inplace=True)
    return df
