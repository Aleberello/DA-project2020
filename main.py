### Libraries import
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import pickle

from utils.utils import *
from utils.roberta import *

import plotly.express as px 

'''
#### CARICAMENTO DATI
## Caricamento datasets
pfizer = pd.read_csv('data/src/vaccination_tweets.csv', parse_dates=[4,9], infer_datetime_format=True)
vacc = pd.read_csv('data/src/covidvaccine.csv', parse_dates=[3,8], infer_datetime_format=True)

## Join dei dataset
comb = pd.concat([pfizer[vacc.columns], vacc], ignore_index=True)

## Rimozione eventuali duplicati basandosi su text, user_name e date
data = comb.drop_duplicates(subset=['user_name','date','text'], keep='first', ignore_index=True)

#### ANALISI PRELIMINARE DATI RAW
#data.head()
#data.shape
#data.dtypes
data.info()

#### DATA CLEANING
## Gestione valori nulli
data.isnull().sum()
data = data.dropna(subset=['user_name'])
data = data.fillna('None')

## Standardizzazione locations
data['country'] = get_countries(data['user_location'])

## Interpretazione array hashtags
data['hashtags'] = data.hashtags.apply(eval)

## Salvataggio dati
data.to_pickle('./data/data.pkl')

'''


#### SENTIMENT EXTRACTION
data = pd.read_pickle("data/data.pkl")

## VADER Sentiment extraction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

def vader_preproc(text):
    '''
    Preprocessing basico del testo.
    Dato che il metodo è sensibile a caps, emoji, punteggiatura, slang ed in generale a messaggi di tipo "social"
    vengono solo rimossi gli url, le menzioni e gli hashtags.
    '''
    new_text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', text) #urls
    new_text = re.sub(r'(?:@[\w_]+)', '', new_text) #mentions
    new_text = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', new_text) #hash-tags
    new_text = re.sub(r'[\n\r\t]', '', new_text) #char escapes
    return new_text.strip()

def get_vader_sent(score):
    '''
    Estrae la label del sentiment dallo score compound estratto dall'analyzer Vader, le soglie sono quelle consigliate
    dagli autori del metodo.
    '''
    if score>=0.05:
        return 'Positive'
    elif score<=(-0.05):
        return 'Negative'
    else:
        return 'Neutral'


vsent = pd.DataFrame(data['text'])
vsent['text-clean'] = vsent['text'].apply(lambda x : vader_preproc(x))
vsent['score'] = vsent['text-clean'].apply(lambda x : vader_analyzer.polarity_scores(x)['compound'])
vsent['sentiment'] = vsent['score'].apply(get_vader_sent)

vsent.to_pickle('./data/vader_sent.pkl')


## Twitter-RoBERTa Sentiment Extraction
texts = pd.DataFrame(data['text'])
rsent = get_roberta_sentiment(texts)

rsent.to_pickle('./data/roberta_sent.pkl')








'''
## BERTtweet embedding and K-Means
import torch
from transformers import AutoModel, AutoTokenizer 

#model_name="vinai/bertweet-base"
model_name="vinai/bertweet-covid19-base-cased"
bertweet = AutoModel.from_pretrained(model_name)

# Con il campo normalization viene applicata la normalizzazione definita dagli autori, visualizzabile anche qua:
# https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)

import pdb; pdb.set_trace()

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples

'''