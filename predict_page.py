import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scholarly import scholarly
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
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


# Define a function to classify input text and recommend similar articles
def classify_text(text):
    preprocessed_text = gensim.utils.simple_preprocess(text)
    bow_vector = dictionary.doc2bow(preprocessed_text)
    topic_probs = lda_model.get_document_topics(bow_vector)
    topic_probs = [x[1] for x in topic_probs]
    X_tfidf = tfidf_vectorizer.transform([text])
    topic_label = clf.predict(X_tfidf)[0]

    # Find similar articles
    query = scholarly.search_pubs(topic_label)
    recommendations = []
    i = 0
    while i < 3:
        try:
            publication = next(query)
            if publication and 'bib' in publication and publication['bib'].get(
                    'abstract'):
                doi = publication['bib'].get('doi', 'N/A')
                recommendations.append((publication['bib']['title'],
                                        publication['bib']['abstract'], doi))
                i += 1
        except StopIteration:
            break

    return topic_label, recommendations


# Define the Streamlit app
def predict_page():
    st.title('Topic Classification and Recommendation App')
    st.write(
        'This app allows you to classify a piece of text into one of the topics in our dataset and receive recommendations for similar articles.'
    )
    view_dataset = st.checkbox('View dataset')

    # If the checkbox is checked, display the dataset
    if view_dataset:
        st.write('Displaying dataset...')
        st.write(final_df.head(10))

    # Create a text input for the user to enter their text
    user_input = st.text_input('Enter your text here:')

    # Add a checkbox to allow the user to view the dataset

    # Classify the user's input and display the topic label and recommendations when they click the "Classify" button
    if st.button('Classify'):
        st.write("These are the recommended topics from google scholar: ")
        topic_label, recommendations = classify_text(user_input)
        st.write(f'Topic label: {topic_label}')
        if recommendations:
            for i, (title, abstract, doi) in enumerate(recommendations):
                st.write(f"{i + 1}. Title: {title}\n")
                st.write(f'Abstract: {abstract}\n')
                st.write(f'Link/DOI: {doi}\n')
        else:
            st.write("No recommendations found.")
    # PyLDAvis visualization
        vis = gensimvis.prepare(lda_model, bow_corpus, dictionary)
        pyLDAvis.display(vis)


# Run the Streamlit app
if __name__ == '__main__':
    predict_page()