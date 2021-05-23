import json
import logging
import sys

import click
from typing import Tuple

from ml_project.data import read_data, split_train_test_data

from ml_project.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params
)

from ml_project.features import FeaturesExtractor, extract_target

from ml_project.models import (
    train_model,
    predict_model, 
    evaluate_model, 
    save_model, 
    save_metrics,
    load_model
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def train_pipeline(training_pipeline_params: TrainingPipelineParams) -> Tuple[str, dict]:
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)

    train_df, test_df = split_train_test_data(
        data, training_pipeline_params.splitting_params
    )

    feature_extractor = FeaturesExtractor(training_pipeline_params.feature_params)
    
    train_features = feature_extractor.fit_transform(train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")
    logger.info("features and target created")
    
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    test_features = feature_extractor.transform(test_df)
    test_target = extract_target(test_df, training_pipeline_params.feature_params)

    logger.info(f"test_features.shape is {test_features.shape}")
    predicts = predict_model(model, test_features)

    metrics = evaluate_model(predicts, test_target)
    save_metrics(metrics, training_pipeline_params.metric_path)
    
    path_to_model = save_model(model, training_pipeline_params.output_model_path)
    logger.info(f"metrics is {metrics}")
    return path_to_model, metrics
        
        
@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()