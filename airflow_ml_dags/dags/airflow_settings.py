from datetime import timedelta

VOLUMES = ['/Users/ruakgvn/PycharmProjects/MADE/krveronika/airflow_ml_dags/data/:/data']

DEFAULT_ARGS = {
    "owner": "Kruglikova Veronika",
    "email": ["krveronika0191@gmail.com"],
    "email_on_failure": True, # Alert в случае падения дага
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "max_active_runs": 1
}

SENSOR_ARGS = {
    "poke_interval":10,
    "timeout": 60,
    "mode": "reschedule"
}

DATA_DIR = "data/raw"
MODEL_DIR = "data/model"
OUTPUT_DIR = "data/output"
