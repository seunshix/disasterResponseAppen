# Disaster Response Pipeline Project

### Introduction
This project is designed to analyze and categorize messages sent during a disaster. The goal is to help disaster response organizations quickly identify the needs of people affected by the disaster, so they can provide the right support at the right time.

### Project Components
The Disaster Response Pipeline project consists of the following components:

- An ETL (Extract, Transform, Load) pipeline that processes and cleans the raw disaster message data.
- A machine learning pipeline that trains a model to categorize messages into various categories.
- A web app that presents the results of the model and allows users to input their own disaster-related messages to be categorized.

### Getting Started
1. Clone the repository
```
$ git clone https://github.com/<username>/disaster_response_pipeline.git
$ cd disaster_response_pipeline
```
2. Run the ETL pipeline

```
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
3. Run the machine learning pipeline
```
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
4. Run the web app
```
$ python app/run.py
```
5. Open a web browser and go to http://0.0.0.0:3001/ to view the web app.

### File Descriptions
- data/process_data.py: The ETL pipeline script that processes and cleans the disaster message data.
- models/train_classifier.py: The machine learning pipeline script that trains a model to categorize messages.
- app/run.py: The script that runs the Flask web app that presents the results of the model.
- app/templates: A folder that contains the HTML templates used by the Flask web app.
- data/disaster_messages.csv: The raw disaster message data.
- data/disaster_categories.csv: The raw disaster categories data.
- data/DisasterResponse.db: The SQLite database that stores the cleaned disaster message data.
- models/classifier.pkl: The pickled machine learning model that categorizes disaster messages.

# Libraries and Tools Used
- pandas
- nltk
- scikit-learn
- sqlalchemy
- plotly
- flask

### Acknowledgements
This project is part of the Udacity Data Scientist Nanodegree program. The disaster message data is provided by Figure Eight.
