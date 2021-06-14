import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from airflow_settings import VOLUMES, DEFAULT_ARGS, SENSOR_ARGS

with DAG(
        "train_model",
        default_args=DEFAULT_ARGS,
        schedule_interval="@weekly",
        start_date=days_ago(10),
) as dag:

    data_sensor = FileSensor(
        task_id = "data_sensor",
        filepath = "data/raw/{{ ds }}/data.csv",
        **SENSOR_ARGS
    )
    
    target_sensor = FileSensor(
        task_id = "target_sensor",
        filepath = "data/raw/{{ ds }}/target.csv",
        **SENSOR_ARGS
    )
    
    preprocess = DockerOperator(
        task_id = "data_processing",
        image = "airflow-preprocess",
        command = "/data/raw/{{ ds }} /data/processed/{{ ds }} /data/model/{{ ds }}",
        network_mode = "bridge",
        do_xcom_push = False,
        volumes = VOLUMES,
    )
    
    split = DockerOperator(
        task_id = "data_splitting",
        image = "airflow-split",
        command = "/data/processed/{{ ds }} /data/splitted/{{ ds }}",
        network_mode = "bridge",
        do_xcom_push = False,
        volumes = VOLUMES
    )

    train = DockerOperator(
        task_id = "model_training",
        image = "airflow-train",
        command = "/data/splitted/{{ ds }} /data/model/{{ ds }}",
        network_mode = "bridge",
        do_xcom_push = False,
        volumes = VOLUMES
    )
    
    validate = DockerOperator(
        task_id = "model_validation",
        image = "airflow-validate",
        command = "/data/splitted/{{ ds }} /data/model/{{ ds }}",
        network_mode = "bridge",
        do_xcom_push = False,
        volumes = VOLUMES
    )

    [data_sensor, target_sensor] >> preprocess >> split >> train >> validate
    
