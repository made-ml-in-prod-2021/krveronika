import sys
import yaml
import click
import pandas as pd
from ml_project.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params
)

from ml_project.data import read_data
from ml_project.models import predict_model, load_model
from ml_project.features import FeaturesExtractor
import logging.config

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def predict_pipeline(
        training_pipeline_params: TrainingPipelineParams,
):
    model = load_model(training_pipeline_params.output_model_path)
    logger.info(f"loaded model from {training_pipeline_params.output_model_path} for prediction")
    df = read_data(training_pipeline_params.input_data_path)
    extracted_features = FeaturesExtractor(training_pipeline_params.feature_params).fit_transform(df)
    logger.info(f"features extracted; extracted_features size: {extracted_features.shape}")
    predict = predict_model(model, extracted_features)
    logger.info(f"prediction done; prediction size: {predict.shape}")
    pd.DataFrame(predict, columns=['target']).to_csv(training_pipeline_params.output_predict_path, index=False)



@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()