import sys
import nltk
import pandas as pd
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine


nltk.download('all')


def load_data(database_filepath):
    '''
    Function
    Load the data from database

    Input
    database_filepath: path of the database file

    Output
    X (dataframe) : dataframe containing the feature variable
    Y (dataframe) : dataframe containing target variables
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'] # feature variable
    Y = df.iloc[:, 4:] # target variable

    return X, Y




def tokenize(text):
    '''
    Function
    split the sentence into individual words and return its basic form

    Input
    text (str) : sentence to be tokenized

    Output
    lemmas (list of str) : list containing base form of text

    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Remove punctuation, special characters and spaces
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    
    # Perform lemmatization on the tokens
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        
    return lemmas



def build_model():
    '''
    Function
    Builds a model that classifies disaster messages

    Output
    cv(list of str): classification model

    '''

    # Create a pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tdidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  
    ])

    # Specify parameters for Grid Search
    parameters = {
    'tdidf__use_idf' : (True, False),
    'clf__estimator__n_estimators': [50, 100, 200]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid= parameters, cv = 3, n_jobs = 1)

    return cv

def evaluate_model(model, X_test, y_test):
    '''
    Function
    Evaluates the classification model and prints out evaluation scores
    
    Input
    X_test : test feature
    y_test : test target
    

    Output
        Prints out f1 score, precision, recall and accuracy of the pipeline
    '''

    y_pred = model.predict(X_test)
    for i in range(36): # 36 is the number of columns in y_test
        print(f'Feature {i + 1}: {y_test.columns[i]}')
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division  = 0))
        print(60 * '-')

    # calculate accuracy
    accuracy = (y_pred == y_test.values).mean()
    print(f'Accuracy : {accuracy * 100}')


def save_model(model, model_filepath):
    '''
    Function
    Saves a pickle file of the model

    Input
    model: classification model created
    model_filepath (str): path of the pickle file

    Output
    Pickle file
    '''
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()