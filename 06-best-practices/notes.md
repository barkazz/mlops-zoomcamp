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

