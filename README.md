# Disaster Response Pipeline Project

During a disaster, it can be challenging for organizations to quickly identify the needs of affected people and provide them with the right support. The Disaster Response Pipeline project aims to solve this problem by analyzing and categorizing messages sent during a disaster.

The task is to develop an end-to-end pipeline that can extract, transform, and load raw disaster message data, train a machine learning model to categorize messages into various categories, and present the results of the model through a user-friendly web app.

The project consists of three main components. The first component is an ETL pipeline that cleans and preprocesses raw disaster message data. The second component is a machine learning pipeline that uses natural language processing techniques to train a model to categorize messages into various categories. The third component is a web app that presents the results of the model and allows users to input their own disaster-related messages to be categorized.

The Disaster Response Pipeline project has been successful in developing an end-to-end pipeline that can effectively categorize disaster-related messages. The project can be useful in helping disaster response organizations quickly identify the needs of affected people and provide them with the right support at the right time.


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

### Improvements
1. Enhance the user interface of the website to make it more user-friendly and easier to access.
2. Gather additional disaster datasets to improve the machine learning model and enable more accurate classification of messages. This will help in expanding the scope of the project and making it more robust.

### Acknowledgements
This project is part of the Udacity Data Scientist Nanodegree program. The disaster message data is provided by Figure Eight.

