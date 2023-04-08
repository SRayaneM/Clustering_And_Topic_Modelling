from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

app = Flask(__name__)

# Load the LDA model
lda_model = LdaModel.load('Main/model.pkl')

# Load the dictionary
dictionary = Dictionary.load('Main/model.pkl')


# Define a function to preprocess the text
def preprocess_text(text):
    processed_text = simple_preprocess(text, deacc=True, min_len=2, max_len=15)
    return processed_text


# Define a function to get the topic probabilities for an abstract
def get_topic_probabilities(abstract):
    processed_abstract = preprocess_text(abstract)
    bow_abstract = dictionary.doc2bow(processed_abstract)
    topic_probabilities = lda_model.get_document_topics(bow_abstract)
    return topic_probabilities


# Define a function to get the top 3 topics with their probabilities
def get_top_topics(topic_probabilities, num_topics=3):
    top_topics = sorted(topic_probabilities, key=lambda x: x[1],
                        reverse=True)[:num_topics]
    return top_topics


# Define a function to categorize the type of research article topic
def categorize_topic(top_topics):
    topic_labels = [
        'Maths', 'Physics', 'Computer Science', 'Statistics',
        'Quantitative Biology', 'Quantitative Finance'
    ]
    topic_probabilities = np.zeros(len(topic_labels))
    for topic in top_topics:
        topic_probabilities[topic[0]] = topic[1]
    normalized_probabilities = topic_probabilities / topic_probabilities.sum()
    category = topic_labels[np.argmax(normalized_probabilities)]
    return category, normalized_probabilities


# Define the Flask routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    abstract = request.form['abstract']
    topic_probabilities = get_topic_probabilities(abstract)
    top_topics = get_top_topics(topic_probabilities)
    category, normalized_probabilities = categorize_topic(top_topics)
    return render_template('index.html',
                           category=category,
                           probabilities=normalized_probabilities)


if __name__ == '__main__':
    app.run(debug=True)
