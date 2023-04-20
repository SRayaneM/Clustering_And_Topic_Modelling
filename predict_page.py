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

st.sidebar.success("Select a page")

header = st.container()
with header:
    st.title('Research Paper Abstract Recommendation System')

    text1 = '<p style="font-family:Courier; font-size: 24px;">An abstract is a short summary of your completed research. It is intended to describe your work without going into great detail.</p>'
    st.markdown(text1, unsafe_allow_html=True)

    #Importing the data set
    data = pd.read_csv('Main/abstracts.csv')
    train_df = data.head(10000)

    #### TF-IDF
    TF_IDF = TfidfVectorizer()
    TF_IDF_ = TF_IDF.fit_transform(train_df['ABSTRACT'])
    df_tf = pd.DataFrame(TF_IDF_.toarray(),
                         columns=TF_IDF.get_feature_names_out())

    #### Cosine Similarity
    sim = pd.DataFrame(cosine_similarity(df_tf, df_tf))

    #### Function 1
    def recommend(Abstract):
        abs_id = train_df[(train_df.ABSTRACT == Abstract)][
            train_df['id']].values[0]  #getting the id of the abstract
        scores = list(
            enumerate(sim[abs_id])
        )  #getting the corresponding sim values for input abstract
        sorted_scores = sorted(scores, key=lambda x: x[1],
                               reverse=True)  # Sorting sim values
        sorted_scores = sorted_scores[1:]
        abstracts = [
            train_df[abstracts[0] == train_df['id']]['ABSTRACT'].values[0]
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

chromedriver = "Main\chromedriver.exe"
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
