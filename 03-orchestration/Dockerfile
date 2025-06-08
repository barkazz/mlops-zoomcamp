FROM apache/airflow:2.7.3

# copy your requirements
COPY requirements.txt .

# install Airflow at exactly the same version,
# then your extras from requirements.txt
RUN pip install \
      apache-airflow==${AIRFLOW_VERSION} \
    && pip install \
      -r requirements.txt \



# FROM apache/airflow:2.7.3

# USER root
# RUN apt-get update && apt-get install -y gcc python3-dev libpq-dev
# USER airflow

# RUN pip install --no-cache-dir \
#     mlflow==2.9.2 \
#     scikit-learn==1.2.2 \
#     pandas==2.0.3 \
#     hyperopt==0.2.7 \
#     xgboost==2.0.0 \
#     fastparquet==2023.4.0 \
#     boto3==1.28.57 \
#     pyarrow==14.0.2 \
#     requests==2.31.0