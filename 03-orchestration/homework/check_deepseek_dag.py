from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import pickle
import os
import requests
from pathlib import Path

# Constants for file paths
DATA_DIR = "/data"
TRAIN_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet"
TEST_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30)
}

def download_parquet(url: str, file_path: Path):
    """Download a parquet file with retry logic"""
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded: {file_path}")
            return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt == 2:
                raise

def download_data():
    """Downloads and saves train/test datasets"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download train data (February 2023)
    train_path = Path(DATA_DIR) / "yellow_tripdata_2023-02.parquet"
    download_parquet(TRAIN_URL, train_path)
    
    # Download test data (March 2023)
    test_path = Path(DATA_DIR) / "yellow_tripdata_2023-03.parquet"
    download_parquet(TEST_URL, test_path)

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to process a single dataframe"""
    # Filter invalid records
    df = df[(df['trip_distance'] > 0) & (df['passenger_count'] > 0)].copy()
    
    # Calculate duration
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filter trips between 1-60 minutes
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    
    # Create categorical features
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    # Extract time features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.dayofweek
    
    return df[['duration', 'PU_DO', 'trip_distance', 'pickup_hour', 'pickup_day']]

def preprocess_data():
    """Preprocesses data and saves results"""
    # Read and process train data
    train_df = pd.read_parquet(Path(DATA_DIR) / "yellow_tripdata_2023-02.parquet")
    train_df = process_dataframe(train_df)
    train_df.to_parquet(Path(DATA_DIR) / "preprocessed_train.parquet")
    
    # Read and process test data
    test_df = pd.read_parquet(Path(DATA_DIR) / "yellow_tripdata_2023-03.parquet")
    test_df = process_dataframe(test_df)
    test_df.to_parquet(Path(DATA_DIR) / "preprocessed_test.parquet")

def train_model():
    """Trains model and saves artifacts"""
    train_df = pd.read_parquet(Path(DATA_DIR) / "preprocessed_train.parquet")
    
    # Prepare features
    categorical = ['PU_DO', 'pickup_hour', 'pickup_day']
    numerical = ['trip_distance']
    train_dicts = train_df[categorical + numerical].to_dict(orient='records')
    
    # Train vectorizer and model
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = train_df['duration'].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Save artifacts
    with open(Path(DATA_DIR) / 'vectorizer.b', 'wb') as f:
        pickle.dump(dv, f)
    
    with open(Path(DATA_DIR) / 'model.b', 'wb') as f:
        pickle.dump(lr, f)
    
    print(f"Model trained with {len(y_train)} samples")

def test_model():
    """Tests model and outputs RMSE"""
    test_df = pd.read_parquet(Path(DATA_DIR) / "preprocessed_test.parquet")
    
    # Load artifacts
    with open(Path(DATA_DIR) / 'vectorizer.b', 'rb') as f:
        dv = pickle.load(f)
    
    with open(Path(DATA_DIR) / 'model.b', 'rb') as f:
        lr = pickle.load(f)
    
    # Prepare features
    categorical = ['PU_DO', 'pickup_hour', 'pickup_day']
    numerical = ['trip_distance']
    test_dicts = test_df[categorical + numerical].to_dict(orient='records')
    
    # Generate predictions
    X_test = dv.transform(test_dicts)
    y_pred = lr.predict(X_test)
    
    # Calculate and log RMSE
    rmse = np.sqrt(mean_squared_error(test_df['duration'], y_pred))
    print(f"Test RMSE: {rmse:.2f} minutes")
    
    # Save results
    with open(Path(DATA_DIR) / 'rmse.txt', 'w') as f:
        f.write(f"{rmse:.4f}")

with DAG(
    'duration_prediction',
    default_args=default_args,
    description='Taxi Duration Prediction Pipeline',
    schedule_interval='@monthly',
    start_date=datetime(2023, 2, 1),
    catchup=False,
    tags=['mlops'],
    max_active_runs=1
) as dag:
    
    download_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data
    )
    
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )
    
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
    
    test_task = PythonOperator(
        task_id='test_model',
        python_callable=test_model
    )
    
    download_task >> preprocess_task >> train_task >> test_task