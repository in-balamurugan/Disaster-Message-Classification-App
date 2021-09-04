# Disaster Response Pipeline Project
A machine learning pipeline to categorize emergency messages based on the needs communicated by the sender and served in a web app built with flask. 

## Table of Contents

1. [Folders](#Folders)
2. [Dependecies](#Dependecies)
3. [Instructions](#Instructions)
4. [Web App Screen shots](#App)


<a name="Folders"></a>
### Folders:
--&nbsp; app<br>
|    &nbsp; &nbsp;&nbsp;  -- templates<br>
|    &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- master.html  # main page of web app<br>
|    &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- go.html # classification result page of web app<br>
|-- run.py # Flask file that runs app<br>
|--data<br>
|    &nbsp; &nbsp;&nbsp;  -- disaster_categories.csv  # data to process<br>
|    &nbsp; &nbsp;&nbsp;  -- disaster_messages.csv  # data to process<br>
|    &nbsp; &nbsp;&nbsp;  -- process_data.py<br>
|    &nbsp; &nbsp;&nbsp;  -- DisasterResponse.db   # database to save clean data<br>
|--models<br>
|    &nbsp; &nbsp;&nbsp;  -- train_classifier.py <br>
|    &nbsp; &nbsp;&nbsp;  -- classifier.pkl  # saved model <br>
|    &nbsp; &nbsp;&nbsp;  -- classifier.pkl  # saved model <br>
|    &nbsp; &nbsp;&nbsp;  -- classifier.pkl  # saved model <br>
|    &nbsp; &nbsp;&nbsp;  -- classifier.pkl  # saved model <br>
|--screen_shots<br>
--&nbsp; README.md<br>


<a name="Dependecies"></a>
### Package Dependecies:
* nltk
* flask
* sqlalchemy
* sklearn
* joblib
* icecream
* pandas


<a name="Instructions"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.<br>
> To run ETL pipeline that cleans data and store in database<br>`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`<br><br>
>   To run ML pipeline that trains classifier and save<br> `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    
2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

<a name="app"></a>
### Web App Screen shots:

Welcome screen

<img src="/screen_shots/welcome.png" alt="welcome" width="80%" height="80%"/>

Enter message classification
<img src="/screen_shots/enter_message.png" alt="enter message" width="80%" height="80%"/>

Classification results page
<img src="/screen_shots/results.png" alt="results" width="80%" height="80%"/>


