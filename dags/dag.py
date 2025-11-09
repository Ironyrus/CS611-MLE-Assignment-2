from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline
    # --- label store ---
    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id='run_gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # --- training and inference ---
    run_train_and_inference = BashOperator(
        task_id='run_train_and_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_train.py '
            '--train_start_date 2023-01-01 '
            '--train_end_date {{ ds }} '
        ),
    )

    run_visualization = BashOperator(
        task_id='run_visuals',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 visualize_perf.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Define task dependencies to run scripts sequentially
    bronze_label_store >> silver_label_store >> gold_label_store >> run_train_and_inference >> run_visualization