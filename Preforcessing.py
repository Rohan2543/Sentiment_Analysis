import sqlite3
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, pairwise_distances
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tqdm import tqdm
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

pd.options.mode.chained_assignment = None


def data_StopWordsCleaning(df):
    # Remove Stop Words
    stop = stopwords.words('english')
    f = lambda x: ' '.join([word for word in x.split() if word not in (stop)])
    df['review'] = df['review'].dropna().apply(f)
    return df


def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(str(text))]

def lemmatize(s):
     s = [wnl.lemmatize(word) for word in s]
     return s
    
def data_TokenizationLemmatization(df):
    tokenizer = RegexpTokenizer(r'\w+')
    df['review'] = df['review'].dropna().apply(
        lambda x: ' '.join(word for word in tokenizer.tokenize(x)))
    
    # Lowercase Words
    #wnl = WordNetLemmatizer()
    #f = lambda x: ' '.join([word for word in x.split() if word not in (wnl)])
    #df['reviewContent'] = df['reviewContent'].dropna().apply(f)

    df['review'] = df['review'].dropna().apply(
        lambda x: x.lower())
    
    df['review'] = df.review.apply(lemmatize_text)

    #df['reviewContent'] =str(df['reviewContent'])
    df['review'] = df['review'].apply(lambda x: ','.join(map(str, x)))
    
    #df['reviewContent'] = df['reviewContent'].map(lambda x: x.lstrip('+-').rstrip('\''))
    #df['reviewContent'] = df['reviewContent'].map(lambda x: x.lstrip('+-').rstrip(','))
    #df['reviewContent'] = df['reviewContent'].map(lambda x: x.lstrip('+-').rstrip(']'))
    #df['reviewContent'] = df['reviewContent'].map(lambda x: x.lstrip('+-').rstrip('['))
    
    #df["reviewContent"]=df["reviewContent"].str.replace("'","")
    #df["reviewContent"]=df["reviewContent"].str.replace(",","")
    #df["reviewContent"]=df["reviewContent"].str.replace("]","")
    #df["reviewContent"]=df["reviewContent"].str.replace("[","")
    
    return df

    
def data_cleaning(df):
    print("Cleaning Data")
    # Removing emtpy cells
    if len(np.where(pd.isnull(df))) > 2:
        # TODO
        pass
    print("Data Cleaning Complete")
    return df


