### Libraries import
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import pickle
import plotly.express as px 

from utils.utils import *
from utils.roberta import *


"""

#### Caricamento dati
## Caricamento datasets
pfizer = pd.read_csv('datasets/vaccination_tweets.csv', parse_dates=[4,9], infer_datetime_format=True)
vacc = pd.read_csv('datasets/covidvaccine.csv', parse_dates=[3,8], infer_datetime_format=True)

## Join dei dataset
comb = pd.concat([pfizer[vacc.columns], vacc], ignore_index=True)

## Rimozione eventuali duplicati basandosi su text, user_name e date
data = comb.drop_duplicates(subset=['user_name','date','text'], keep='first', ignore_index=True)

#### Analisi preliminare dati raw
data.head()
data.shape
data.dtypes
data.info()

#### Data cleaning
## Gestione valori nulli
data.isnull().sum()
#px.imshow(data.isnull()).show() # Visualizzazione heatmap nulli
data = data.dropna(subset=['user_name'])
data = data.fillna('None')

## Standardizzazione locations
data['country'] = get_countries(data['user_location'])

## Salvataggio dati
data.to_pickle('./bkp/data.pkl')

"""
#### EDA
data = pd.read_pickle("data.pkl")



#### Sentiment Analysis
'''
## VADER Sentiment extraction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def vader_preproc(text):
    new_text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', text) #urls
    new_text = re.sub(r'(?:@[\w_]+)', '', new_text) #mentions
    new_text = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', new_text) # hash-tags
    return new_text.strip()

analyser.polarity_scores(vader_preproc(data.text[0]))
'''


## Twitter-RoBERTa-sentiment extraction
#get_roberta_sentiment(data.text)






import pdb; pdb.set_trace()




## BERTtweet embedding
import torch
from transformers import AutoModel, AutoTokenizer 
#import tensorflow as tf
#tf.get_logger().setLevel(logging.ERROR)

model_name="vinai/bertweet-base"
#model_name="vinai/bertweet-covid19-base-cased"
bertweet = AutoModel.from_pretrained(model_name)
# Con il campo normalization viene applicata la normalizzazione definita dagli autori, visualizzabile anche qua:
# https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)

import pdb; pdb.set_trace()
# INPUT TWEET IS ALREADY NORMALIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples
