"""
Project:  Disaster Response Pipeline
Script:   ETL Pipeline
Function: Cleaning a dataset of text messages 
Author:   Norman Hofer
"""

# ------------------------------------------------------------------- Import Packages ------------------------------------------------------------------------------

import pandas as pd
from sqlalchemy import create_engine
import sys

# ---------------------------------------------------------------------- Load Data ---------------------------------------------------------------------------------

def load_data(message_data_path, categorie_data_path):
    """
    Load message and categorie Data
    
    Arguments:
        - message_data_path: filepath of csv-file with message data
        - categorie_data_path: filepath of csv-file with category data
    Output:
        - df: Dataframe with loaded and merged message and category data
    """
    
    # Load message and category data    
    messages = pd.read_csv(message_data_path)
    categories = pd.read_csv(categorie_data_path)
    
    # merge datasets
    df = messages.merge(categories, how='left', on='id')
    
    return df

# ---------------------------------------------------------------------- Clean Data --------------------------------------------------------------------------------

def clean_data(df):
    """
    Clean data
    
    Arguments:
        - df: Dataframe with loaded and merged message and category data
    Output:
        - df_cleaned: Cleaned dataframe with loaded and merged message and category data
    """

    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # Use the first row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df_cleaned = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df_cleaned.drop_duplicates(inplace=True)
    
    return df_cleaned

# -------------------------------------------------------------------- Save Cleaned Data -------------------------------------------------------------------------------

def save_data(df, db):
    """
    Save cleaned data
    
    Arguments:
        - df: Cleaned dataframe 
        - db: database name
    """
    
    engine = create_engine('sqlite:///' + db)
    df.to_sql('df', engine, index=False)
    pass

# -------------------------------------------------------------------------- Main --------------------------------------------------------------------------------------
    
def main():
    """
    Main processing function
    
    Execution of the following tasks:
        1. Load Data
        2. Process Data
        3. Save Cleaned Data to SQlite Database
    """
    
    print('ETL Pipeline started.')
    
    if len(sys.argv) == 4:
        message_data_path, categorie_data_path, db = sys.argv[1:]
    
        # 1. Load Data
        print('Loading Data...')
        df = load_data(message_data_path, categorie_data_path)
        
        # 2. Process Data
        print('Cleaning Data...')
        df_cleaned = clean_data(df)
        
        # 1. Save Data
        print('Saving Cleaned Data...')
        save_data(df_cleaned, db)
        
        print('Cleaned Data saved to Database succesfully!')
 
# Start
if __name__ == '__main__':
    main()