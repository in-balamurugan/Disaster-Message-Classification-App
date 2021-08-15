# import libraries
import sys
import pandas as pd
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import nltk.data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
#from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
#from sklearn.metrics import multilabel_confusion_matrix
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from icecream import ic
from datetime import datetime
import pickle



def time_format():
    return f'{datetime.now()}|>'

#Configs
ic.configureOutput(prefix=time_format,includeContext=True)

def load_data(database_filepath,DEBUG=0):
    conn = sqlite3.connect(database_filepath)
    cur = conn.cursor()
    ic(cur)
    df = pd.read_sql("SELECT * FROM messages", con=conn)
    conn.commit()
    conn.close()
    if DEBUG == True:
        df=df.head(10)
    X = df['message']
    Y = df.iloc[:,4:]
    categories= list(Y.columns)
    return X,Y,categories


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('starting_verb', StartingVerbExtractor()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [5, 5],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
 
    #for Y_test,Y_pred in zip(Y_test.values,Y_pred):
     #   if Y_test.all()== Y_pred.all():
      #      ic()
       #     correct_predictions += 1
    #
    #accuracy=correct_predictions / len(Y_test)
    #print("Accuracy:", accuracy)
    
    
    accuracy = (Y_pred == Y_test).mean()

    
    
    print(accuracy)

    file = open("accuracy.txt", "w")
    file.write("accuracy:" +str(accuracy))
    file.close
    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()