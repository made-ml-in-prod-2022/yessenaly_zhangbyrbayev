import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_project_package.entities.train_params import TrainingParams
from ml_project_package.features.build_features import encode_df
from ml_project_package.entities.feature_params import FeatureParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    df: np.array, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators, random_state=train_params.random_state
        )
    elif train_params.model_type == "LinearRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(df, target)
    return model


def predict(
    model: SklearnClassifierModel, df: np.ndarray
) -> np.ndarray:
    return model.predict(df)


def get_model_quality(
    pred: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, pred),
        "f1": f1_score(target, pred),
        "roc-auc score": roc_auc_score(target, pred),
    }


def predict_unprocessed_df(
    df: pd.DataFrame, model: SklearnClassifierModel, enc: OneHotEncoder, params: FeatureParams
) -> np.ndarray:
    df = encode_df(df, enc, params)
    return predict(model, df)


def save_model(model: object, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str) -> SklearnClassifierModel:
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model