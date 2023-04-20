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

header = st.container()
with header:
    st.title('Research Paper Abstract Recommendation System')

text1 = '<p style="font-family:Courier; font-size: 24px;">An abstract is a short summary of your completed research. It is intended to describe your work without going into great detail.</p>'
st.markdown(text1, unsafe_allow_html=True)