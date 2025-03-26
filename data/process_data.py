import sys
import numpy as np 
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads two dataset from filepaths:
    INPUT: 2 csv files 
    OUTPUT: one merged dataframe 
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=('id'))
    return df
    

def clean_data(df):
    """
    Cleans the merged dataframe:
    INPUT: unprocessed dataframe 
    OUTPUT: cleaned dataframe 
    """
    # 1. Split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = (row.apply(lambda x: x[:-2])).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # 2. Convert category values to just numbers 0 or 1.¶
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # 3. Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    # 4. drop duplicates
    df = df.drop_duplicates(keep='first')
    
    # 5. After a previous check of the df:
    # Variable 'related' -> value '2' looks like a mistake, let´s replace it with the value '1'
    df['related'].replace(2, 1, inplace=True)
    # Variable 'child_alone' has no data -> let´s drop it: 
    df.drop(['child_alone'], axis=1, inplace=True)
    
    return df
    
    
def save_data(df, database_filename):
    """
    Saves dataframe to a sqlite database
    INPUT: cleaned dataframe
    OUTPUT: saved clean datafrme to a sqlite database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Message', engine, if_exists='replace', index=False)


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