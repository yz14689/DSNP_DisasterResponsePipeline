 
# Data Scientist Narodegree - Project : DisasterResponse Pipeline


# In[ ]:


###  
"""
Code 1:  ETL pipeline - process_data.py:
Command to be:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    Function:
    load datafiles (.csv) (specified as parameter)
    merges datafiles
    cleans the merged datafile
    saves the cleaned datafile to sql databse (specified as parameter)
    
    Input:    
    disaster_messages.csv 
    disaster_categories.csv
        
    Output:
    DisasterResponse.db   
    
"""


# In[]:


import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


# In[ ]:

def load_data(messages_filepath, categories_filepath):
    ''' loads the data dir
    Inputs:
        messages_filepath: message.cvs file path
        categories_filepath: category.csv file path
    Output: 
        df: Dataframe
   '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df  = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    '''
    cleans the data
    Input: 
        df: Dataframe (from load_date, category field value as 'related-1;request-0;offer-0; ... ')
        
    '''
    
    # split categories into seperate category name
    categories = df["categories"].str.split(';', expand=True)
    
    # select the first row and extract new column name list
    row = categories.iloc[0]
    
    category_colnames = []
    for x in row:
        category_colnames.append(x[:-2]) 
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string  
        categories[column] = categories[column].str[-1]   
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].astype(int)
        
    # Replace categories column in df with new category columns.
    df.drop(columns=['categories'], inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)
    df.replace(2, 1, inplace = True )
    
    # drops the duplicates
    duplicates= df.duplicated( keep='first')
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
       Saves the dataframe  in sql database
       Input:
        df: dataframe
        database_filenae: filename for the database
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disaster', engine, index = False, if_exists='replace')  # Note the SQL database table name is defined as 'Disaster'


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
              'datasets as the first and second argument respectively, as '  \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()


# In[ ]:
# END




