 
# Data Scientist Narodegree - Project : DisasterResponse Pipeline


# In[ ]:


### 
"""
Code 2:  Machine Learing - train_classifier.py
Command to be:
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

    Function:    
    Load the sql database (from ETL), build a text processing and ML pipeline and multi-output RandomForest classifier  
    Train and tune the model using GridSearch
    Export the trained model as pickle file
    
    Input:    
    SQL database with TABLE named as 'Disaster' (from ETL pipeline code)
        
    Output:
    classifier.pkl
    
"""


# In[ ]:


import sys
import re
import pickle

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score, precision_recall_fscore_support
from sklearn.metrics import make_scorer 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



# In[ ]:

def load_data(database_filepath):
    '''
    load the data from the database in database-filepath (from ETL)
           
    Input: 
        database_filepath (string): Path to database file to export
    output: 
        X (Dataframe): Feature Vector
        Y (Dataframe): Target Vector
        category_names (List): List of strings for column names for categories

    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    # Either the following works, but make sure to specify the SQL table (not database) name as 'Disaster' (from process_data.py)
    #sql =  "SELECT * FROM Disaster"
    #df =  pd.read_sql_table(sql, con = engine)   
    df =  pd.read_sql_query("SELECT * FROM Disaster", con = engine)  
    
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1)
    
    category_names=list(Y.columns)    
    
    return X, Y, category_names


def tokenize(text):
    '''
    tokenize and clean the text 
    Input: 
        text: text to be tokenized and cleaned
    output: 
        clean_tokens: cleaned and tokenized text
    '''
    
    # tokenize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)  
    tokens  = [w for w in tokens if   w not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()  
    
    #iterate through each token
    clean_tokens = []
    for tok in tokens:      
        #lemmatize, normalize case, and remove leading/trailing white space
        clean_tok =  lemmatizer.lemmatize(tok).lower().strip( )
        clean_tokens.append(clean_tok)   
        
    return clean_tokens


def build_model():
    ''' 
    building the model 
    
    output: 
        (model): Pipeline and gridsearch model
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier(random_state=100)))
         ])
    
    # hyperparameters
    parameters = {'clf__estimator__n_estimators': [50, 100] ,
                  'clf__estimator__min_samples_split': [2, 4]
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters, return_train_score=True, verbose=2) # n_jobs=1, cv=1 by default
    # The verbose is to print out the porgress, he higher the verbose the more information
    # This step took very long - abour 6hrs, but can be revised to the following to improve the efficienty:
    # cv = GridSearchCV(pipeline, param_grid=parameters, return_train_score=True, verbose=2, n_jobs=4) 

    return cv


def calc_scores(y_test, y_pred, category_names):
    '''
    Inputs:
        y_test: testing labels
        y_pred: predicted labels
        category_names: names of the labels     
    '''
    
    res =[]
    precision =0
    recall=0
    f1score =0
    for i in range(len(category_names)):
         res = (precision_recall_fscore_support(y_test.iloc[:,i].values, y_pred[:,i], average='weighted'))
         precision += res[0]
         recall += res[1]
         f1score += res[2]
     
    precision = precision/len(category_names)
    recall = recall/len(category_names)
    f1score = f1score/len(category_names)
    
    tot_accuracy = (y_pred == y_test).mean().mean()
    
    print('Average Weighted Prediction Scores:')
    print ("Precision: {:2f} Recall: {:2f}  F1-Score: {:2f}".format(precision*100, recall*100, f1score*100))
    print('total Accuracy: %2.2f'% (tot_accuracy*100))
    

def evaluate_model(model, X_test, y_test, category_names):
    '''  
    Function returns the performance of test set for each category_names
   
    inputs: 
        model: trained model
        X_test: testing data
        y_test: data labels
        category_names: names of the labels
    '''
    y_pred = model.predict(X_test)
    calc_scores(y_test, y_pred, category_names)
    

def save_model(model, model_filepath):
    '''
    save the trained moel in python pickle file
    model: trained model
    model_filepath: where the model is located
    '''
    pickle.dump(model, open(model_filepath, "wb" ) )


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to '  \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# In[ ]:
# END