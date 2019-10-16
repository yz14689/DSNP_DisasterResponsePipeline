# Project Name 
Disaster Response Pipeline (Udecity Data Science Narodegree)

# Required packages


# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# TO DO LIST


## ETL pipeline
### process.py
    Load the datafiles, merge and clean the combined data, save the cleaned data to SQL database for subsequent use


## ML pipeline
### train_classifer.py
    Load the sql database (from ETL), build a text processing and ML pipeline and multi-output RandomForest classifier. 
    Train and tune the model using GridSearch
    Export the trained model as pickle file


## final execution 
### run.py 


## Screenshot of output
