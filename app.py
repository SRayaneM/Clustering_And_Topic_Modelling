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

# Train a random forest classifier
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform([x[0] for x in lda_data])
y = [x[1] for x in lda_data]
clf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tfidf, y)

# Load another dataset for article recommendations
recommendation_data = pd.read_csv('Main\Model\mtrain.csv')
recommendation_tfidf = tfidf_vectorizer.transform(
    recommendation_data['ABSTRACT'])


# Define a function to classify input text and recommend similar articles
def classify_text(text):
    preprocessed_text = gensim.utils.simple_preprocess(text)
    bow_vector = dictionary.doc2bow(preprocessed_text)
    topic_probs = lda_model.get_document_topics(bow_vector)
    topic_probs = [x[1] for x in topic_probs]
    X_tfidf = tfidf_vectorizer.transform([text])
    topic_label = clf.predict(X_tfidf)[0]

    # Find similar articles
    doc_similarities = cosine_similarity(X_tfidf, recommendation_tfidf)
    doc_similarities_sorted = sorted(enumerate(doc_similarities[0]),
                                     key=lambda x: x[1],
                                     reverse=True)
    top_docs = [x[0] for x in doc_similarities_sorted][:3]
    recommendations = []
    for doc_index in top_docs:
        title = recommendation_data.iloc[doc_index]['TITLE']
        abstract = recommendation_data.iloc[doc_index]['ABSTRACT']
        recommendations.append((title, abstract))

    return topic_label, recommendations


# Define the Streamlit app
def app():
    st.title('Topic Classification and Recommendation App')
    st.write(
        'This app allows you to classify a piece of text into one of the topics in our dataset and receive recommendations for similar articles.'
    )

    # Create a text input for the user to enter their text
    user_input = st.text_input('Enter your text here:')

    # Classify the user's input and display the topic label and recommendations when they click the "Classify" button
    if st.button('Classify'):
        topic_label, recommendations = classify_text(user_input)
        st.write(f'Topic label: {topic_label}')
        st.write('Recommended articles:')
        for i, (title, abstract) in enumerate(recommendations):
            st.write(f'{i+1}. {title}')
            st.write(f'{abstract}\n')


# Run the Streamlit app
if __name__ == '__main__':
    app()