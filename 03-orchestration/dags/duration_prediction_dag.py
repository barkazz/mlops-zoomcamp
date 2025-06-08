from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb

#import mlflow

#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_experiment("nyc-taxi-experiment")

#models_folder = Path('models')
#models_folder.mkdir(exist_ok=True)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='duration_prediction_dag',
    description='Airflow DAG to train and evaluate taxi ride duration prediction model',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    def extract_data():
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-3.parquet'
        df = pd.read_parquet(url)
        print(df.shape[0], 'rows extracted')

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        print(df.shape[0], 'rows after filtering')

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)

        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

        return df


    # Define tasks
    t1 = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )

"""    t2 = PythonOperator(
        task_id='prepare_features',
        python_callable=prepare_features
    )

    t3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    t4 = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate
    )"""

    # Task dependencies
t1 #>> t2 >> t3 >> t4