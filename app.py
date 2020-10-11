
import streamlit as st
import pickle
#import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re as re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem import WordNetLemmatizer


def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result
def remove_punct(review):
    review_new="".join([c for c in review if c not in string.punctuation]).lower()
    review_new=re.sub('[0-9]+', '',review_new)
    return review_new
def remove_stop_words(tokens):
    words=[t for t in tokens if t not in stopwords.words('english')]
    return words
def word_lemmitizer(tokens):
    lemmatizer=WordNetLemmatizer()
    token_new=[lemmatizer.lemmatize(i) for i in tokens]
    return token_new

#x=x.str.join('')
#x.head()

#print(x)
model=pickle.load(open("movie_review_classifier","rb"))
#model=loaded_model(open("movie_review_classifier.h5","rb"))
st.title('Movie Review Sentiment Analyzer')
review=st.text_area('Enter Review','Type Here....')

review=remove_tags(review)
review=remove_punct(review)
review=word_tokenize(review)
review=remove_stop_words(review)
review=word_lemmitizer(review)
review=pd.Series(' '.join(review))
prediction=model.predict(review)
if st.button("Predict"):
  st.title(prediction)
