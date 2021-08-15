import sys
import pandas as pd
from icecream import ic
from datetime import datetime
from sqlalchemy import create_engine


def time_format():
    return f'{datetime.now()}|>'


# Configs
ic.configureOutput(prefix=time_format, includeContext=True)


def load_data(messages_filepath, categories_filepath):
    """"
    Loads and merges Messages and their categories
    Parameters:
    messages_filepath(string)
    categories_filepath(string)
    Returns:    
    df(dataframe)
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df


def clean_data(df):
    """"
    Extracts each column, sets column name and extracts cell values
    eg:  'request:0' will be with column name 'request' and row value of '0'
    Parameters:
    df(dataframe)
    Returns:
    df(dataframe)
    """

    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[1]
    category_colnames = list(map(lambda x: x[:-2], row))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    # convert column from string to numeric
    categories = categories.astype(int)

    # drop the original categories column from `df`
    frames = [df, categories]
    df = pd.concat([df, categories], axis=1)
    df.drop(['categories'], axis=1, inplace=True)

    # Drop the duplicates.
    df = df[~df.duplicated()]

    return df


def save_data(df, database_filename):
    """"
    save the dataframe to sqlite db
    Parameters:
    df(dataframe)
    Returns:
    -
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()