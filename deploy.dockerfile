# Note: To run this dockerfile you will need to have the 
# datasets/formatted_bbb_df.csv file in you datasets folder

FROM ubuntu

RUN apt update

RUN apt -y install python3-dev python3-pip

COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

COPY python_modules /python_modules/

WORKDIR /python_modules/

CMD ["python3", "flask_app.py"]

# Build with: docker build -f deploy.dockerfile -t cricket_rating_app_deploy .

# Run with: docker run -it -p 5010:5050 cricket_rating_app_deploy
