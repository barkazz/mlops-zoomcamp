


#1

mkdir output

python batch.py 2023 3



#2

pipenv install --dev pytest

To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.

mkdir tests



#3

pipenv run pytest tests


You have a few options to see your print output:

Run pytest with -s
This turns off output capturing, so you’ll see all your print statements as the tests run:

pipenv run pytest -s tests



#4

pipenv install --dev awscli

docker-compose up -d

docker ps

docker-compose down

aws s3 mb s3://nyc-duration

http://127.0.0.1:4566/

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"

aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

aws --endpoint-url=http://localhost:4566 s3 ls



#5

###pipenv install aiobotocore==2.4.2
pipenv install awscli botocore

# start Localstack if needed:
docker run --rm -d -p 4566:4566 localstack/localstack



export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export S3_ENDPOINT_URL='http://localhost:4566'

export AWS_EC2_METADATA_DISABLED=true
export AWS_DISABLE_IMDS=true

python integration_test.py


aws --endpoint-url=http://localhost:4566 s3 ls

