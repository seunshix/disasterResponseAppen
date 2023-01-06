import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function:
        Load the data from the csv files, merge both into
        one dataframe and return the dataframe

    Input:
        messages_filepath (str):
            file path of messages csv file
        categories_filepath (str):
            file path of categories csv file
    
    Output:
        df (Datframe):
            Dataframe of messages and categories merged
    
    '''
    messages = pd.read_csv(messages_filepath) # load messages dataset
    categories = pd.read_csv(categories_filepath) # load categories dataset

    df = pd.merge(messages, categories, how = 'inner', on = 'id') # merge datasets

    return df
    


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)

    row = categories.iloc[2].tolist() # select the first row of the categories dataframe
    category_colnames = [x[:-2] for x in row]  

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert the columns to numbers(0, 1)
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 0)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()