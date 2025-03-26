# import standard libraries
import re 
import pickle 
import sys

# import libraries
import nltk
import numpy as np 
import pandas as pd 
from sqlalchemy import create_engine


#nltk downloads 
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

#nltk modules
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# scikit-learn 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

def load_data(database_filepath):
    """
    Load data from database_filepath
    INPUT: data stored in database_filepath
    OUTPUT: df (loaded dataframe/ X: messages/ y: categories
    """
    
    engine = engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM Message", engine)
    
    #load and view the text and category variables
    X = df.message.values
    y = df.iloc[:, 4:].values
    
    #get categories/column names
    categories = df.columns[4:].tolist()
    
    return X, y, categories

def tokenize(text):
    """
    Tokenize messages
    INPUT: messages
    OUTPUT: clean tokens:
    - normalize (removes punctuation & converts to lowercase)
    - lemmatize
    - tokenize
    """
    
    #Create a function that normalizes, tokenizes and lemmatizes the messages
    #remove punctuation such as Twitter(#) and tags(@) and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    
    #Split text into words using NLTK
    tokens = word_tokenize(text) 
    
    #Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #Remove stopwords and lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words("english")
    ]
    
    return clean_tokens
    

def build_model():
    """
    Build a machine learning pipeline using scikit-learn & GridSearchCV
    INPUT: message variable
    OUTPUT: classifies results in 36 categories
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1))), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    
    # Specify the parameters for grid search 
    parameters = {
        # Try different n-gram ranges (1 word)
        'vect__ngram_range': [(1, 1)],
        # Try different number of trees in the random forest 
        'clf__estimator__n_estimators': [5, 10]
    }
    
    # Create a grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the results of the previously built model using classification_report()
    INPUT: prediction based on test data
    OUTPUT: print classification_report()
    """
    #Predict on test data (transform test data)
    y_pred = model.predict(X_test)
    
    #Print classification report for all categories 
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save model into a pickle file
    INPUT: built model
    OUTPUT: pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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