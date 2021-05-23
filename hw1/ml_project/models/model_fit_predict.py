import json
import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from ml_project.enities.train_params import TrainingParams


SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, 
    target: pd.Series, 
    train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, 
            min_samples_leaf=5,
            random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, 
    features: pd.DataFrame, 
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, 
    target: pd.Series
) -> Dict[str, float]:
    scores= {
        "roc_auc_score": roc_auc_score(target, predicts),
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }
    return scores


def save_model(model: SklearnRegressionModel, path: str) -> str:
    """ save model to pickle file """
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def save_metrics(metrics: dict, path: str):
    """ save metrics to json """
    with open(path, 'w') as fout:
        json.dump(metrics, fout)
        
        
def load_model(path: str) -> SklearnRegressionModel:
    """ load model from pickle file """
    with open(path, 'rb') as fin:
        load_model = pickle.load(fin)
    return load_model

