from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
    schedule_interval='0 9 15 * *',  # run on the 15th of every month at 09:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    params={'year': 2023, 'month': 3}
) as dag:

    def extract_data(**context):
        year = context['params']['year']
        month = context['params']['month']
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)
        # Compute duration
        df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
        # Filter outliers
        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
        os.makedirs('/tmp/data', exist_ok=True)
        input_path = f'/tmp/data/{year}-{month:02d}-raw.parquet'
        df.to_parquet(input_path, index=False)
        context['ti'].xcom_push(key='raw_path', value=input_path)

    def prepare_features(**context):
        raw_path = context['ti'].xcom_pull(key='raw_path')
        df = pd.read_parquet(raw_path)
        df['PULocationID'] = df['PULocationID'].astype(str)
        df['DOLocationID'] = df['DOLocationID'].astype(str)
        train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        y_train = df['duration'].values
        # Persist features and vectorizer
        os.makedirs('/tmp/features', exist_ok=True)
        feat_path = '/tmp/features/X_train.npz'
        dv_path = '/tmp/features/dv.bin'
        pickle.dump(dv, open(dv_path, 'wb'))
        # Using scipy to save sparse matrix
        from scipy import sparse
        sparse.save_npz(feat_path, X_train)
        context['ti'].xcom_push(key='feat_path', value=feat_path)
        context['ti'].xcom_push(key='dv_path', value=dv_path)
        context['ti'].xcom_push(key='y_path', value='/tmp/features/y_train.pkl')
        pickle.dump(y_train, open('/tmp/features/y_train.pkl', 'wb'))

    def train_model(**context):
        feat_path = context['ti'].xcom_pull(key='feat_path')
        y_path = context['ti'].xcom_pull(key='y_path')
        X_train = sparse.load_npz(feat_path)
        y_train = pickle.load(open(y_path, 'rb'))
        model = LinearRegression()
        model.fit(X_train, y_train)
        os.makedirs('/tmp/model', exist_ok=True)
        model_path = '/tmp/model/linreg.bin'
        pickle.dump(model, open(model_path, 'wb'))
        context['ti'].xcom_push(key='model_path', value=model_path)

    def evaluate(**context):
        feat_path = context['ti'].xcom_pull(key='feat_path')
        y_path = context['ti'].xcom_pull(key='y_path')
        model_path = context['ti'].xcom_pull(key='model_path')
        X_train = sparse.load_npz(feat_path)
        y_train = pickle.load(open(y_path, 'rb'))
        model = pickle.load(open(model_path, 'rb'))
        y_pred = model.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        print(f"RMSE: {rmse}")

    # Define tasks
    t1 = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )

    t2 = PythonOperator(
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
    )

    # Task dependencies
    t1 >> t2 >> t3 >> t4
