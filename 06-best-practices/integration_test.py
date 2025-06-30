# integration_test.py
import os
import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime

import batch  # your batch.py from the repo

def dt(hour, minute, second=0):
    # Now pretending it's January 2023
    return datetime(2023, 1, 1, hour, minute, second)

# Localstack endpoint
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

# Create S3 client
s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'test'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'test'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
)

# your 4 rows from Q3, but on Jan 2023
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1,    1,    dt(1, 2), dt(1, 10)),
    (1,   None, dt(1, 2, 0), dt(1, 2, 59)),
    (3,    4,    dt(1, 2, 0), dt(2, 2, 1)),
]
columns = [
    'PULocationID',
    'DOLocationID',
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime'
]
df_input = pd.DataFrame(data, columns=columns)

# Get S3 paths
input_uri = batch.get_input_path(2023, 1)
output_uri = batch.get_output_path(2023, 1)

# Parse bucket and key
bucket = input_uri.split('/')[2]
key = '/'.join(input_uri.split('/')[3:])

# Create in-memory Parquet file
buffer = BytesIO()
df_input.to_parquet(buffer, engine='pyarrow', compression=None, index=False)
buffer.seek(0)

# Create bucket if not exists
try:
    s3.head_bucket(Bucket=bucket)
except s3.exceptions.ClientError:
    s3.create_bucket(Bucket=bucket)

# Upload to S3
s3.upload_fileobj(buffer, bucket, key)
print(f"Wrote test data to {input_uri}")