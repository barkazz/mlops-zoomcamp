from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from sklearn.model_selection import train_test_split

#import os
#import pickle

#from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
#import xgboost as xgb

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("homework-3-nyc-taxi")

#models_folder = Path('models')
#models_folder.mkdir(exist_ok=True)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
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
        # Load only necessary columns to reduce memory
        cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
        url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
        df = pd.read_parquet(url, columns=cols)

        print(f"{len(df):,} rows extracted")

        # vectorized duration in minutes
        df['duration'] = (
            df.tpep_dropoff_datetime
            .sub(df.tpep_pickup_datetime)
            .dt
            .total_seconds()
            .div(60)
        )

        # filter outliers
        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
        #df = df[(df.duration >= 1) & (df.duration <= 60)]
        print(f"{len(df):,} rows after filtering")

        # categorical features
        df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
        
        #for col in ('PULocationID', 'DOLocationID'):
        #    df[col] = df[col].astype(str)
        #df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

        return df


    def df_to_dict(df):
        return df[categorical].to_dict(orient='records')

    def prepare_features(df):

        df_clean['target'] = df_clean['duration']
        categorical = ['PULocationID', 'DOLocationID']

        train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)
        dv = DictVectorizer()
        X_train = dv.fit_transform(df_to_dict(train_df))
        X_val = dv.transform(df_to_dict(val_df))
        y_train = train_df['target'].values
        y_val = val_df['target'].values

        with mlflow.start_run():
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="models",
            input_example=X_val[:1],
            signature=mlflow.models.signature.infer_signature(X_val, y_pred)
            )


    # Define tasks
    extract_data_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )

    prepare_features_task = PythonOperator(
        task_id='prepare_features',
        python_callable=prepare_features
    )
    """
    t3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    t4 = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate
    )"""

    # Task dependencies
extract_data_task >> prepare_features_task #>> t3 >> t4