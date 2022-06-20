from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(10),
    "email": ["airflow@example.com"],
    "retries": 0,
    "retry_delay": timedelta(days=1),
}

with DAG(
    dag_id="predict_dag",
    default_args=default_args,
    description="A DAG for prediction",
    schedule_interval=timedelta(days=1),
) as dag:
    predict = BashOperator(
        task_id="predict",
        bash_command=f"python /opt/airflow/scripts/predict.py -od /opt/airflow/data/predictions/",
    )
    predict