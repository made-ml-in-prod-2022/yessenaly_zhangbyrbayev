import sys
from time import sleep
import logging
import json
import argparse

from ml_project_package.data.make_dataset import (
    read_data, 
    split_train_val_data, 
)
from ml_project_package.entities.train_pipeline_params import (
    read_training_pipeline_params, 
    TrainingPipelineParams, 
)
from ml_project_package.features.build_features import (
    extract_target,
    build_encoder, 
    encode_df,
    save_encoder,
)
from ml_project_package.models.model_functions import (
    train_model, 
    get_model_quality, 
    predict_unprocessed_df,
    save_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def train_pipeline(config_path: str) -> None:
    """"""
    training_pipeline_params = read_training_pipeline_params(config_path)
    return run_train_pipeline(training_pipeline_params)

def run_train_pipeline(training_pipeline_params: TrainingPipelineParams) -> None:
    """"""
    logger.info("start train pipeline with params {}".format(training_pipeline_params))
    data = read_data(training_pipeline_params.input_data_path)
    logger.info("data.shape is {}".format(data.shape))
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    train_df = train_df.drop([training_pipeline_params.feature_params.target_col], axis=1)
    val_df = val_df.drop([training_pipeline_params.feature_params.target_col], axis=1)

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")
    enc = build_encoder(train_df, training_pipeline_params.feature_params)
    train_df_encoded = encode_df(train_df, enc, training_pipeline_params.feature_params)
    logger.info(f"train_df_encoded.shape is {train_df_encoded.shape}")
    model = train_model(
        train_df_encoded, train_target, training_pipeline_params.train_params
    )

    val_pred = predict_unprocessed_df(
        val_df,
        model, 
        enc, 
        training_pipeline_params.feature_params,
    )
    metrics = get_model_quality(
        val_pred,
        val_target,
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    save_encoder(enc, training_pipeline_params.output_encoder_path)
    
    path_to_model = save_model(
        model, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    train_pipeline(args.cp)