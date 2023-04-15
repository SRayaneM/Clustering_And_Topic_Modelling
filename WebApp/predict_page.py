import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests
import time, os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from selenium.common.exceptions import NoSuchElementException


import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests
import time, os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from selenium.common.exceptions import NoSuchElementException

st.set_page_config(layout="wide")

header = st.container()
with header:
    st.title('Research Paper Abstract Recommendation System')

text1 = '<p style="font-family:Courier; font-size: 24px;">An abstract is a short summary of your completed research. It is intended to describe your work without going into great detail.</p>'
st.markdown(text1, unsafe_allow_html=True)

# st.text('An abstract is a short summary of your completed research. It is intended to describe your work without going into great detail.')

#Turing machines and G\"odel numbers are important pillars of the theory of computation. Thus, any computational architecture needs to show how it could relate to Turing machines and how stable implementations of Turing computation are possible. In this chapter, we implement universal Turing computation in a neural field environment. To this end, we employ the canonical symbologram representation of a Turing machine obtained from a G\"odel encoding of its symbolic repertoire and generalized shifts. The resulting nonlinear dynamical automaton (NDA) is a piecewise affine-linear map acting on the unit square that is partitioned into rectangular domains. Instead of looking at point dynamics in phase space, we then consider functional dynamics of probability distributions functions (p.d.f.s) over phase space. This is generally described by a Frobenius-Perron integral transformation that can be regarded as a neural field equation over the unit square as feature space of a dynamic field theory (DFT). Solving the Frobenius-Perron equation yields that uniform p.d.f.s with rectangular support are mapped onto uniform p.d.f.s with rectangular support, again. We call the resulting representation \emph{dynamic field automaton}.

############	st.markdown()

code_container = st.container()
with code_container:
    st.title('')
    # st.text('Enter the abstract of a paper you found insightful so we can help you find papers as useful:')
    text2 = '<p style="font-family:Courier; font-size: 24px;">Enter the abstract of a paper you found insightful so we can help you find relevant papers who are as useful:</p>'
    st.markdown(text2, unsafe_allow_html=True)
    #### importing data
    data = pd.read_csv('abstract.csv')
    sample = data.head(10000)

    #### TF-IDF
    CV_TF_IDF = TfidfVectorizer()
    CV_TF_IDF_ = CV_TF_IDF.fit_transform(sample['ABSTRACT'])
    df_tf = pd.DataFrame(CV_TF_IDF_.toarray(),
                         columns=CV_TF_IDF.get_feature_names())

    #### Cosine Similarity
    sim = pd.DataFrame(cosine_similarity(df_tf, df_tf))

    #### Function 1
    def recommend(Abstract):
        abs_id = sample[(sample.abstract == Abstract
                         )]['id'].values[0]  #getting the id of the abstract
        scores = list(
            enumerate(sim[abs_id])
        )  #getting the corresponding sim values for input abstract
        sorted_scores = sorted(scores, key=lambda x: x[1],
                               reverse=True)  # Sorting sim values
        sorted_scores = sorted_scores[1:]
        abstracts = [
            sample[abstracts[0] == sample['id']]['abstract'].values[0]
            for abstracts in sorted_scores
        ]
        return abstracts

    #### Function 2
    def recommend_3(abstract_list):
        first_3 = []
        count = 0
        for abstract in abstract_list:
            if count > 2:
                break
            count += 1
            first_3.append(abstract)

        return first_3

    #### prompting input
    sel_col, disp_col = st.columns(2)
    my_abstract = sel_col.text_input('Enter your abstract:', '')
    list_ = recommend(my_abstract)
    recommendations = recommend_3(list_)

chromedriver = "/Users/saraali/Downloads/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
chromeOptions = webdriver.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images": 2}
chromeOptions.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(chromedriver, chrome_options=chromeOptions)
driver.get(
    "https://www.google.com/search?q=m&sxsrf=AOaemvJMpDSDsv17E3QxjxLdOqymXdlF-w%3A1636528339453&source=hp&ei=03CLYa6xGNqQ9u8Pi_easAw&iflsig=ALs-wAMAAAAAYYt-4y-vV258A6wGAFwH3mzAyuz4bXnn&oq=m&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECcyBAgjECcyBAgjECcyCwgAEIAEELEDEIMBMggIABCABBCxAzIFCAAQsQMyBQgAELEDMgsIABCABBCxAxCDATIICC4QgAQQsQMyCAguELEDEIMBOgcIIxDqAhAnUIYDWIYDYM0FaAFwAHgAgAGjAYgBowGSAQMwLjGYAQCgAQGwAQo&sclient=gws-wiz&ved=0ahUKEwju5taSn430AhVaiP0HHYu7BsYQ4dUDCAY&uact=5"
)

for i in range(len(recommendations)):

    search_box = driver.find_element_by_xpath(
        '/html/body/div[4]/div[2]/form/div[1]/div[1]/div[2]/div/div[2]/input'
    )  # search bar xpath
    #clear the current search
    search_box.clear()
    #input new search
    search_box.send_keys(recommendations[i])  # abstract
    #hit enter
    search_box.send_keys(Keys.RETURN)
    time.sleep(1)

    try:
        title = driver.find_element_by_xpath(
            '//*[@id="rso"]/div[1]/div/div/div[1]/a/h3/span')
        link = driver.find_element_by_xpath(
            '//*[@id="rso"]/div[1]/div/div/div[1]/a')
        st.write(title.text, '\n\n')
        st.write(link.get_attribute("href"), '\n\n')
        st.write(recommendations[i], '\n\n\n\n')

    except NoSuchElementException:

        st.write('\n', 'Search Google More Thoroughly', '\n\n')
        st.write('Search Google More Thoroughly', '\n\n')
        st.write(recommendations[i], '\n\n\n')
