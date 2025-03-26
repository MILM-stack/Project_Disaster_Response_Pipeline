import re, sys
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

#As per train_classifier.py, create a function that normalizes, tokenizes and lemmatizes the messages
def tokenize(text):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponseProject.db')
df = pd.read_sql("SELECT * FROM Message", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = df.columns[4:].tolist()    
       
    top_10_categories = category_counts.nlargest(10)
    cat_name = top_10_categories.index.tolist()   
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #Visualization 1: Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        #Visualization 2: Distribution of Message by Category
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        #Visualization 3: Top 10 Message Categories 
        {
            'data': [
                Bar(
                    x=cat_name,
                    y=top_10_categories
                )
            ],

            'layout': {
                'title': 'Top 10 Message Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top 10"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()