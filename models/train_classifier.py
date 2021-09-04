# import libraries
import sys
import pandas as pd
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from icecream import ic
from datetime import datetime
import pickle
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import numpy as np


def time_format():
    return f'{datetime.now()}|>'


# Configs
ic.configureOutput(prefix=time_format, includeContext=True)


def load_data(database_filepath, DEBUG=0):
    """"
    Load cleaned data from sqlite
    Parameters:
    database_filepath(string)
    DEBUG(Boolean, to toggle between debugging and training)
    Returns:
    df(dataframe)
    """
    conn = sqlite3.connect(database_filepath)
    cur = conn.cursor()
    ic(cur)
    df = pd.read_sql("SELECT * FROM messages", con=conn)
    conn.commit()
    conn.close()
    if DEBUG:
        df = df.head(100)

    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    X = df['message']
    Y = df.iloc[:, 4:]
    categories = list(Y.columns)
    return X, Y, categories


def tokenize(text):
    """"
    Convert message into tokens removing stopwords and lematising
    Parameters:
    text(string)
    Returns:
    lemmed(string)
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    """"
    Convert text to token counts, does TfIDTransformer and Multioutput class
    Parameters:
    text(string)
    Returns:
    cv(model)
    """
    xgb_clf = xgb.XGBClassifier(learning_rate=0.1,
                                n_estimators=3000,
                                max_depth=10,
                                min_child_weight=1,
                                subsample=0.8,
                                objective='multi:softmax',
                                nthread=4,
                                num_class=36,
                                seed=27
                                )

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=27)))
        # ('clf', MultiOutputClassifier(estimator=xgb_clf))
    ])

    # A parameter grid for XGBoost
    ic(pipeline.get_params().keys())
    # Number of trees in random forest
    n_estimators = [200]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [10, 50, 100]
    # Minimum number of samples required to split a node
    min_samples_split = [2]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    criterion = ['entropy', 'gini']
    weights = [1, 10, 25, 50, 75, 99, 100, 1000]

    parameters = {'clf__estimator__n_estimators': n_estimators,
                  'clf__estimator__min_samples_split': min_samples_split,
                  # 'clf__estimator__criterion': criterion,
                  # 'clf__estimator__max_depth': max_depth,
                  # 'clf__estimator__min_samples_leaf': min_samples_leaf,
                  # 'clf__estimator__bootstrap':bootstrap,
                  # 'clf__estimator__scale_pos_weight':weights
                  }

    ic(parameters)

    cross_valid = KFold(10, True, 1)
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=cross_valid, scoring='f1_micro',
                      verbose=0)
    ic(cv)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """"
    Evaluates model based on accuracy,f1, recall, precision
    Parameters:
    text(string)
    Returns:
    cv(model)
    """
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()

    pd.DataFrame(X_test).to_csv("models/X_test.csv")
    pd.DataFrame(Y_pred).to_csv("models/Y_pred.csv")
    pd.DataFrame(Y_test).to_csv("models/Y_test.csv")

    precision = precision_score(Y_pred, Y_test, average='micro', labels=np.unique(Y_pred))
    recall = recall_score(Y_pred, Y_test, average='micro', labels=np.unique(Y_pred))
    f1 = f1_score(Y_pred, Y_test, average='micro')

    ic(precision)
    ic(recall)
    ic(f1)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(x_train, y_train)

        ic(model.best_params_)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)


    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
