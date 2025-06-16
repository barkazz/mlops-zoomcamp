import sys
import pickle
import pandas as pd
import sklearn


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename, columns=cols)
    
    df['duration'] = (
        df.tpep_dropoff_datetime
          .sub(df.tpep_pickup_datetime)
          .dt
          .total_seconds()
          .div(60)
    )

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


year = int(sys.argv[1]) #2023
month = int(sys.argv[2]) #04

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


print('predicted mean duration is', y_pred.mean())


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
