import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import gensim

# Load data and train the LDA model
final_df = pd.read_csv('Main\Abstract_With_topics.csv')

preprocessed_data = [
    gensim.utils.simple_preprocess(doc) for doc in final_df['ABSTRACT']
]
dictionary = gensim.corpora.Dictionary(preprocessed_data)
bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_data]
lda_model = gensim.models.LdaModel(bow_corpus,
                                   num_topics=5,
                                   id2word=dictionary)

# Generate labeled data
lda_data = []
for i, doc in enumerate(preprocessed_data):
    topics = lda_model.get_document_topics(bow_corpus[i])
    topic_probs = [0] * lda_model.num_topics
    for topic in topics:
        topic_probs[topic[0]] = topic[1]
    topic_label = max(range(len(topic_probs)), key=topic_probs.__getitem__)
    lda_data.append((final_df['ABSTRACT'][i], final_df['ABSTRACT_Topic'][i]))

# Train a Multinomial Naive Bayes classifier
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform([x[0] for x in lda_data])
y = [x[1] for x in lda_data]
clf = MultinomialNB().fit(X_tfidf, y)

# Train a Multinomial Naive Bayes classifier
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform([x[0] for x in lda_data])
y = [x[1] for x in lda_data]
clf = RandomForestClassifier().fit(X_tfidf, y)

# Compute document similarities
doc_similarities = cosine_similarity(X_tfidf)

# Define a function to classify input text and recommend similar articles
def classify_text(text):
    preprocessed_text = gensim.utils.simple_preprocess(text)
    bow_vector = dictionary.doc2bow(preprocessed_text)
    topic_probs = lda_model.get_document_topics(bow_vector)
    topic_probs = [x[1] for x in topic_probs]
    X_tfidf = tfidf_vectorizer.transform([text])
    topic_label = clf.predict(X_tfidf)[0]

    # Find similar articles
    doc_index = final_df[final_df['ABSTRACT'] == text].index[0]
    doc_similarities_sorted = sorted(enumerate(doc_similarities[doc_index]),
                                     key=lambda x: x[1],
                                     reverse=True)
    top_docs = [x[0] for x in doc_similarities_sorted if x[0] != doc_index][:3]
    recommendations = final_df.iloc[top_docs]['ABSTRACT'].tolist()

    return topic_label, recommendations


# Define the Streamlit app
def predict_page():
    st.title('Topic Classification and Recommendation App')
    st.write(
        'This app allows you to classify a piece of text into one of the topics in our dataset and receive recommendations for similar articles.'
    )

    # Create a text input for the user to enter their text
    user_input = st.text_input('Enter your text here:')

    # Classify the user's input and display the topic label and recommendations when they click the "Classify" button
    if st.button('Classify'):
        topic_label, recommendations = classify_text(user_input)
        st.write(f'The input text belongs to the {topic_label} topic.')
        st.write('Here are some similar articles:')
        for i, recommendation in enumerate(recommendations):
            st.write(f'{i+1}. {recommendation}')


# Run the Streamlit app
if __name__ == '__main__':
    predict_page()