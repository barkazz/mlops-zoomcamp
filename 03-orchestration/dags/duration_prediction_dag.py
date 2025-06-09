from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("homework-3-nyc-taxi")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id='duration_prediction_dag',
    description='Airflow DAG to train and evaluate taxi ride duration prediction model',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    def extract_data(**kwargs):
        # Load only necessary columns to reduce memory
        cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
        url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
        df = pd.read_parquet(url, columns=cols)

        # compute duration in minutes
        df['duration'] = (
            df.tpep_dropoff_datetime
              .sub(df.tpep_pickup_datetime)
              .dt
              .total_seconds()
              .div(60)
        )

        # filter outliers
        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        # categorical features must be strings
        df['PULocationID'] = df['PULocationID'].astype(str)
        df['DOLocationID'] = df['DOLocationID'].astype(str)

        # push to XCom
        kwargs['ti'].xcom_push(key='df', value=df)
        print(f"Extracted and cleaned {len(df):,} rows")

    def prepare_features(**kwargs):
        ti = kwargs['ti']
        df = ti.xcom_pull(key='df', task_ids='extract_data')

        # set up target & features
        df['target'] = df['duration']
        categorical = ['PULocationID', 'DOLocationID']

        # split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # vectorize
        dv = DictVectorizer()
        train_dicts = train_df[categorical].to_dict(orient='records')
        val_dicts   = val_df[categorical].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        X_val   = dv.transform(val_dicts)

        y_train = train_df['target'].values
        y_val   = val_df['target'].values

        # train & log with MLflow
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

        print(f"Logged LinearRegression model with RMSE={rmse:.3f}")

    extract_data_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True,
    )

    prepare_features_task = PythonOperator(
        task_id='prepare_features',
        python_callable=prepare_features,
        provide_context=True,
    )

    extract_data_task >> prepare_features_task
