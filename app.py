import streamlit as st
import pandas as pd
import nltk
import gensim
import spacy

nltk.download('punkt')
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Load the LDA model
lda_model = gensim.models.ldamodel.LdaModel.load('Main\Model\lda_model.model')

# Load the LDA model
lda_model = LdaModel.load('Main\Model\lda_model.model')

#Load data set
train_df = pd.read_csv('abstracts.csv')


# Define the Streamlit app
def app():
    # Set up the app title and sidebar
    st.title('Research Article Abstract Topic Analysis')
    st.write('Select Input Options')

    # Define the input options
    input_option = st.selectbox('Select Input Option',
                                ['Upload a file', 'Enter text'])

    # Define the file uploader
    if input_option == 'Upload a file':
        uploaded_file = st.file_uploader('Choose a file')
        if uploaded_file is not None:
            abstract = uploaded_file.read()
    else:
        abstract = st.text_area('Enter the Abstract')

    # Define the topic analysis button
    if st.button('Analyze Topics'):
        # Process the abstract
        stop_words = stopwords.words('english')

        #We then create a function to remove the stopwords in our text.
        def remove_stopwords(text):
            text_Array = text.split(' ')
            remove_words = " ".join(
                [i for i in text_Array if i not in stop_words])
            return remove_words

            #And here we will apply the remove_stopwords function. This will remove the stopwords from our dataset's text
            train_df['ABSTRACT'] = train_df['ABSTRACT'].apply(remove_stopwords)

        def lemmatization(texts, allowed_postags=['VERB', 'ADV', 'ADJ']):
            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            output = []
            for sent in texts:
                doc = nlp(sent)
                output.append([
                    token.lemma_ for token in doc
                    if token.pos_ in allowed_postags
                ])
            return output

        text_list = train_df['ABSTRACT'].tolist()

        tokenized_reviews = lemmatization(text_list)

        # Get the topics and associated probabilities
        topics = lda_model.get_document_topics(tokenized_reviews)

        # Display the results
        st.write('Topic Analysis Results:')
            num_topics = lda_model.num_topics
            for i, topic in enumerate(topics):
                if i >= num_topics:
                    break

                topic_id = topic[0]
                topic_prob = topic[1]
                if topic_id < num_topics:
                    st.write(
                        f'Topic {topic_id}: {topic_prob:.3f} - {lda_model.print_topic(topic_id)}'
                    )

        st.write(topic_id)
        st.write(num_topics)


if __name__ == "__main__":
    app()