## Staffery Backend

## Installation

create virtual environment

`python3 -m venv venv`

Then Activate it

`source venv/bin/activate`

Install the requirements

`pip install -r requirements.txt`

## Installing a new package

`pip install package && pip freeze > requirements.txt`

## Migrate the database

`flask db migrate`

## Run the application

`python run.py` OR `export FLASK_APP=run.py` then `flask run`

## Create database on devlopment

`flask db init`

`from app import db`

`db.create_all()`

`db.engine.table_names()`

`exit()`

## Run On Docker

Build the image:

`docker-compose build`

After the build completes, we can run the container:

`docker-compose up`

To init the DB:

`docker exec -it recrudeo_backend /bin/sh`

then run the commands from `Create database on devlopment`
