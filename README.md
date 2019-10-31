# Project Name 
## Udacity Data Scientist Narodegree - Disaster Response Pipeline

### Required packages (details see requirements.txt)
    sys
    pandas 
    sqlite3
    re
    nltk
    pickle
    numpy 
    json
    plotly
    flask 
    sklearn.
    sqlalchemy 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File used

#### 1. Source data (download from Udecity)
    disaster_messages.csv 
    disaster_categories.csv


#### 2. ETL pipeline
#### process.py
Function:
    Load the datafiles, merge and clean the combined data, save the cleaned data to SQL database for subsequent use

Input:    
    disaster_messages.csv
    disaster_categories.csv
        
Output:
    DisasterResponse.db  


#### 3. ML pipeline
#### train_classifer.py
Function:
    Load the sql database (from ETL), build a text processing and ML pipeline and multi-output RandomForest classifier. 
    Train and tune the model using GridSearch
    Export the trained model as pickle file

Input:    
    SQL database with TABLE named as 'Disaster' (from ETL pipeline code)
        
Output:
    classifier.pkl
 

#### 4. Execution code 
#### run.py 
Function:    
    Execute the builtup web app
     
Input:    
    SQL database 'DisasterResponse.db' with table named as 'Disaster' (from ELT code)
    classifier.pkl (from ML pipeline code)
    Web supporting files: go.html and master.html 
        
Output:
    Web app page
    
    (Note:
    The master.html is updated with the desired information to be shown on the final web page. These files are saved in folder 'templates', as the folder name is defined in flask doc)

#### 5. Go to: http://0.0.0.0:3001/
#### This command is for working in Udecity workspace. Given I am working on local PC, I use http://localhost:3001 as web output.

Function:    
    Present the content of pre-defined web app, including the charts and the online user-entered message classifier function


### Screenshot of output

See the file named as "DSNP Disaster Response Pipeline_Output_to github.docx".

