# hpo.py
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer

# 1. Функция для загрузки и предобработки
def read_and_prep(path):
    df = pd.read_parquet(path)
    df['duration'] = (
        df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[['PULocationID', 'DOLocationID']] = (
        df[['PULocationID', 'DOLocationID']]
        .astype(str)
    )
    return df

# 2. Загрузка train/val
df_train = read_and_prep('data/green_tripdata_2023-01.parquet')
df_val   = read_and_prep('data/green_tripdata_2023-02.parquet')

# 3. Подготовка признаков
train_dicts = df_train[['PULocationID', 'DOLocationID']].to_dict('records')
val_dicts   = df_val[['PULocationID', 'DOLocationID']].to_dict('records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
X_val   = dv.transform(val_dicts)

y_train = df_train.duration.values
y_val   = df_val.duration.values

# 4. Пространство гиперпараметров
search_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
    'max_depth':    hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf':  hp.quniform('min_samples_leaf', 1, 4, 1),
    'random_state': 42
}

# 5. Целевая функция для hyperopt
def objective(params):
    # Преобразуем параметры из float в int, где нужно
    params['n_estimators']      = int(params['n_estimators'])
    params['max_depth']         = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf']  = int(params['min_samples_leaf'])
    
    with mlflow.start_run():
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        
        # Логгируем в MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
    
    return {'loss': rmse, 'status': STATUS_OK}

# 6. Запуск оптимизации
if __name__ == "__main__":
    mlflow.set_experiment("random-forest-hyperopt")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,    # можно увеличить
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    print("=== Best hyperparameters ===")
    # Преобразуем полученные “лучшие” гиперпараметры в читаемый вид
    best_params = {
        'n_estimators':      int(best['n_estimators']),
        'max_depth':         int(best['max_depth']),
        'min_samples_split': int(best['min_samples_split']),
        'min_samples_leaf':  int(best['min_samples_leaf']),
        'random_state':      42
    }
    print(best_params)
