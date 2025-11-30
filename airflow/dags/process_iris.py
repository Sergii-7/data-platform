from __future__ import annotations

import os
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

from dbt_operator import DbtOperator
from python_scripts.train_model import process_iris_data

# У контейнері стара назва таймзони
KYIV_TZ = pendulum.timezone("Europe/Kiev")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

# Налаштування dbt-проєкту homework
ANALYTICS_DB = os.getenv("ANALYTICS_DB", "analytics")
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
HOMEWORK_PROJECT_DIR = f"{AIRFLOW_HOME}/dags/dbt/homework"
DBT_PROFILE = "homework"

env_vars = {
    "ANALYTICS_DB": ANALYTICS_DB,
    "DBT_PROFILE": DBT_PROFILE,
}

dbt_vars = {
    "is_test": False,
    # кожен запуск обробляє свою дату
    "data_date": "{{ ds }}",
}

with DAG(
    dag_id="process_iris",
    description="HW: dbt Iris transformations + ML training",
    default_args=default_args,
    # 3 дні: 2025-04-22..2025-04-24
    start_date=KYIV_TZ.datetime(2025, 4, 22, 1, 0, 0),
    end_date=KYIV_TZ.datetime(2025, 4, 24, 23, 59, 59),
    # О 1:00 за Києвом 22–24 квітня
    schedule_interval="0 1 22-24 4 *",
    catchup=True,
    max_active_runs=1,
    tags=["homework", "iris", "dbt", "ml"],
) as dag:
    # 1 dbt seed для iris_dataset
    dbt_seed_iris = DbtOperator(
        task_id="dbt_seed_iris",
        command="seed",
        profile=DBT_PROFILE,
        project_dir=HOMEWORK_PROJECT_DIR,
        env_vars=env_vars,
        vars=dbt_vars,
    )

    # 2 dbt: трансформація Iris
    dbt_run_iris = DbtOperator(
        task_id="dbt_run_iris",
        command="run",
        profile=DBT_PROFILE,
        project_dir=HOMEWORK_PROJECT_DIR,
        env_vars=env_vars,
        vars=dbt_vars,
    )

    # 3 dbt test для контролю якості
    dbt_test_iris = DbtOperator(
        task_id="dbt_test_iris",
        command="test",
        profile=DBT_PROFILE,
        project_dir=HOMEWORK_PROJECT_DIR,
        fail_fast=True,
        env_vars=env_vars,
        vars=dbt_vars,
    )

    # 4 ML: тренування моделі на homework.iris_processed
    train_model = PythonOperator(
        task_id="train_iris_model",
        python_callable=process_iris_data,
        op_kwargs={"process_date": "{{ ds }}"},  # можна й прибрати, якщо не юзаєш
    )

    # 5 просто кінцевий маркер замість email
    notify_success = EmptyOperator(
        task_id="notify_success",
    )

    dbt_seed_iris >> dbt_run_iris >> dbt_test_iris >> train_model >> notify_success