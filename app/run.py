
# Data Scientist Narodegree - Project : DisasterResponse Pipeline


# In[ ]:


### 
"""
Code 3:  final execution  - run.py:
Command to be:
python run.py

Function:    
    Execute the builtup web app
     
    Input:    
    SQL database ('DisasterResponse.db' with table named as 'Disaster', from ELT code)
    classifier.pkl (trained model from ML pipeline code)
        
    Output:
    Model summary

"""


# In[ ]:



import json
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

import plotly
from plotly.graph_objs import Bar

#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


# In[ ]:


#

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# In[ ]:


# load SQL database/table
engine = create_engine('sqlite:///../data/DisasterResponse.db') # specify the SQL database  
df = pd.read_sql_table('Disaster', con = engine) # specify the SQL table (not database) 

# load builtup model
model = joblib.load("../models/classifier.pkl") # specify the trained model package


# In[ ]:



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
   
   # extract data needed for visuals
    # Calculate message count by genre and requested status    
    genre_requested = df[df['request']==1].groupby('genre')['message'].count() 
    genre_not_requested = df[df['request']==0].groupby('genre')['message'].count() 
    genre_names = list(genre_requested.index)
   

    # Calculate message count by genre and death/missing people 
    genre_death = df[df['death']==1].groupby('genre')['message'].count() 
    genre_missing_people = df[df['missing_people']==1].groupby('genre')['message'].count() 
    genre_names = list(genre_death.index) 
    
    
   # create visuals
    graphs = [
        { # data 1
            'data': [
                Bar(
                    x=genre_names,
                    y= genre_requested,  # to be revised
                    name = 'Count of Requested'
               ),
                
                Bar(
                    x=genre_names,
                    y=genre_not_requested, # to be revised
                    name = 'Count of Not Requested'
               )
                
           ],

           'layout': {
               'title': 'Distribution of Message by Genres and Requested',
               'yaxis': {
                   'title': "Count"
               },
               'xaxis': {
                   'title': "Genre"
                } ,
               'barmode': 'group'
            }
        }, # end of data1
        
        {  # data 2
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_death,  # to be revised
                    name = 'Count of Death'
               ) ,
                
                Bar(
                    x=genre_names,
                    y=genre_missing_people,  # to be revised
                    name = 'Count of Missing People'
               ) ,
                
                
                
           ],

           'layout': {
               'title': 'Distribution of Message by Genres and Dealth/Missing People',
               'yaxis': {
                   'title': "Count"
               },
               'xaxis': {
                   'title': "Genre"
                },
               'barmode': 'group'
            }
        } # end of data2
        
        
    ]
   
   
   # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
   
   # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# In[ ]:


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()


# In[ ]:
# END
