import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

from ml_project_package.entities.feature_params import FeatureParams

def build_categorical_encoder(categorical_df: pd.DataFrame) -> OneHotEncoder:
    """"""
    enc = OneHotEncoder()
    enc.fit(categorical_df)
    return enc

def build_encoder(df: pd.DataFrame, params: FeatureParams) -> OneHotEncoder:
    """"""
    categorical_df = df[params.categorical_features]
    return build_categorical_encoder(categorical_df)

def save_encoder(enc: OneHotEncoder, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(enc, f)

def load_encoder(path: str) -> OneHotEncoder:
    with open(path, 'rb') as f:
        enc = pickle.load(f)
    return enc

def encode_categorical_df(categorical_df: pd.DataFrame, enc: OneHotEncoder) -> np.ndarray:
    """"""
    cur = enc.transform(categorical_df)
    return cur.toarray()

def encode_df(df: pd.DataFrame, enc: OneHotEncoder, params: FeatureParams) -> np.ndarray:
    """"""
    categorical_df = df[params.categorical_features]
    numerical_df = df[params.numerical_features]
    categorical_array = encode_categorical_df(categorical_df, enc)
    numerical_array = numerical_df.to_numpy()
    return np.concatenate((numerical_array, categorical_array), axis=1)

def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    """"""
    return df[params.target_col]