"""
Project:  Disaster Response Pipeline
Script:   Machine Learning Pipeline to train a Classifier
Function: Cleaning a dataset of text messages 
Author:   Norman Hofer
"""

# ------------------------------------------------------------------- Import Packages ------------------------------------------------------------------------------

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------- Load Data ---------------------------------------------------------------------------------

def load_data(db):
    """
    Load message and categorie Data from Database
    
    Arguments:
        - db: Database name
    Output:
        - X: Features
        - Y: Labels
        - categories: Category names
    """
    
    # Load data from database
    engine = create_engine('sqlite:///' + db)
    df = pd.read_sql('df', engine)
    df.drop('original', axis=1, inplace=True)
    df.dropna(inplace=True) # drop missing values

    # Separate Features and labels
    X = df.message
    Y = df.iloc[:,3:]
    categories = Y.columns
    
    return X, Y, categories

# ---------------------------------------------------------------------- Tokenize ---------------------------------------------------------------------------------

def tokenize(text):
    """
    Tokenization function to process text data
    
    Arguments:
        - text: Text data to be tokenized
    Output:
        - clean_tokens: Cleaned tokens for Machine Learning
    """
    
    # Detection of urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # Normalization and Tokenizing
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# ------------------------------------------------------------- Model Building ------------------------------------------------------------------------------
    
def build_model():
    """
    Build Machine Learning Model (Classifier)
    
    Output:
        - model: Built Classifier
        
    """
    
    # Model pipline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier())))])
    
    # Setup parameters for Tuning with GridSearch
    parameters = {'clf__estimator__n_estimators': [10,20],
                  'clf__estimator__learning_rate': [1,2]}

    # Parameter Tuning with GridSearch
    model = GridSearchCV(estimator = pipeline, 
                         param_grid = parameters, 
                         n_jobs = -1,
                         verbose = 3)
    
    return model  
    
# ------------------------------------------------------------- Model Evaluation ------------------------------------------------------------------------------
    
def clf_report(model, X_test, y_test, categories):
    '''
    Generation of Classification Report for each Variable
    Input: 
        - model: classifier
        - X_test: test data features
        - y_test: test data labels
        - categories: Names of labels
    Output: 
        - Print of Classification Report
    '''
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = categories)
    for category in categories:
        print('------------------------------------------------------\n')
        print('Classification Report of Feature: {}\n'.format(category))
        print(classification_report(y_test[category],y_pred_df[category]))
    
    pass
  
# ---------------------------------------------------------------- Save Model -----------------------------------------------------------------------------------
    
def save_model(model, model_path):
    '''
    Save Classifier
    Input: 
        - model: classifier
        - path: File path
    '''    
    
    joblib.dump(model, str(model_path) + 'model.pkl')
    pass
    
# -------------------------------------------------------------------------- Main --------------------------------------------------------------------------------------
    
def main():
    """
    Main processing function
    
    Execution of the following tasks:
        1. Load Data
        2. Build Model
        3. Train Model
        4. Evaluate Model
        5. Save trained Model to SQlite Database
    """
    
    print('Machine Learning Pipeline started.')
    
    if len(sys.argv) == 3:
        db, model_path = sys.argv[1:]
    
        # 1. Load Data
        print('Loading Data...')
        X, Y, categories = load_data(db)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        # 2. Build Model
        print('Building Machine Learning Model...')
        model = build_model()

        # 3. Training of Model
        print('Training of Model...')
        model.fit(X_train, y_train)
        
        # 4. Evaluation of Model
        print('Evaluation of Model...')
        clf_report(model, X_test, y_test, categories)
                   
        # 5. Save Model
        print('Saving trained Model...')
        save_model(model, model_path)
        
        print('Trained model saved to Database succesfully!')
 
# Start
if __name__ == '__main__':
    main()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    