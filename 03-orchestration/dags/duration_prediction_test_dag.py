from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os
import pathlib
import pickle

@dag(
    dag_id='taxi_duration_ml_taskflow',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    description='TaskFlow DAG for NYC taxi duration prediction',
    default_args={
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
)
def taxi_duration_pipeline():

    @task
    def extract_data(year_month: str) -> str:
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year_month}.parquet'
        df = pd.read_parquet(url, columns=[
            'tpep_pickup_datetime',
            'tpep_dropoff_datetime',
            'PULocationID',
            'DOLocationID'
        ])

        df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
        df['PULocationID'] = df['PULocationID'].astype(str)
        df['DOLocationID'] = df['DOLocationID'].astype(str)

        out_dir = f'/opt/airflow/data/{year_month}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'cleaned.parquet')
        df.to_parquet(out_path, index=False)

        return out_path

    @task
    def train_model(data_path: str, year_month: str):
        df = pd.read_parquet(data_path)
        df['target'] = df['duration']
        categorical = ['PULocationID', 'DOLocationID']

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_df[categorical].to_dict(orient='records'))
        X_val = dv.transform(val_df[categorical].to_dict(orient='records'))
        y_train = train_df['target'].values
        y_val = val_df['target'].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("nyc-taxi-duration")

        with mlflow.start_run(run_name=f"run-{year_month}"):
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("month", year_month)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="model",
                input_example=X_val[:1],
                signature=mlflow.models.signature.infer_signature(X_val, y_pred)
            )

            # Save DictVectorizer as artifact
            dv_path = pathlib.Path(f"/opt/airflow/data/{year_month}/dv.pkl")
            with open(dv_path, "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(str(dv_path), artifact_path="preprocessor")

        print(f"✅ Model trained for {year_month} with RMSE: {rmse:.2f}")

    # Grab runtime config (e.g., manually set "year_month": "2023-03")
    @task
    def get_config() -> str:
        from airflow.models import Variable
        from airflow.models.param import ParamsDict
        return "{{ dag_run.conf.get('year_month', '2023-01') }}"

    year_month = get_config()
    data_path = extract_data(year_month)
    train_model(data_path, year_month)

taxi_duration_pipeline()
