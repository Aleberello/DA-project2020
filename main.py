#### IMPORT LIBRERIE
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import pickle

from utils.utils import get_countries, get_tokens
from utils.sentiment_extraction import get_vader_sentiment, get_roberta_sentiment



#### CARICAMENTO DATI
## Caricamento datasets
pfizer = pd.read_csv('data/src/vaccination_tweets.csv', parse_dates=[4,9], infer_datetime_format=True)
vacc = pd.read_csv('data/src/covidvaccine.csv', parse_dates=[3,8], infer_datetime_format=True, dayfirst=True)

## Join dei dataset
comb = pd.concat([pfizer[vacc.columns], vacc], ignore_index=True)

## Rimozione eventuali duplicati basandosi su text, user_name e date
data = comb.drop_duplicates(subset=['user_name','date','text'], keep='first', ignore_index=True)


#### ANALISI PRELIMINARE DATI RAW
data.head()
data.shape
data.dtypes
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

## Estrazione tokens dal campo text
data_tokens = pd.DataFrame(data['text'])
data_tokens['tokens'] = data_tokens['text'].apply(get_tokens)

## Salvataggio dati
data.to_pickle('./data/data.pkl')
#data_tokens.to_pickle('./data/data_tokens.pkl')


#### SENTIMENT EXTRACTION
data = pd.read_pickle("data/data.pkl")
texts = pd.DataFrame(data['text'])

## VADER Sentiment extraction
vsent = get_vader_sentiment(texts)
#vsent.to_pickle('./data/vader_sent.pkl')

## Twitter-RoBERTa Sentiment Extraction
rsent = get_roberta_sentiment(texts)
#rsent.to_pickle('./data/roberta_sent.pkl')




'''
#### SVILUPPI FUTURI
## Estrazione BERTtweet sentence embedding e K-Means per identificare cluster e sentiment
import torch
from transformers import AutoModel, AutoTokenizer 

#model_name="vinai/bertweet-base"
model_name="vinai/bertweet-covid19-base-cased"
bertweet = AutoModel.from_pretrained(model_name)

# Con il campo normalization viene applicata la normalizzazione definita dagli autori, visualizzabile anche qua:
# https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples


#### CARICAMENTO DATI AUSILIARI

### Andamento della campagna vaccinale mondiale
vaccinations = pd.read_csv('data/src/pandemy-data/country_vaccinations.csv', 
                            usecols=['country','iso_code','date','total_vaccinations','daily_vaccinations','vaccines'],
                            parse_dates=['date'], infer_datetime_format=True)

## Gestione valori nulli
vaccinations.isnull().sum()
# Gestione valori nulli codice paesi
vaccinations[vaccinations['iso_code'].isnull()].country.unique() # tutti inerenti a stati della Gran Bretagna
vaccinations = vaccinations.dropna(subset=['iso_code'])
vaccinations = vaccinations.fillna(0)

### Andamento dell'epidemia
cases = pd.read_csv('data/src/pandemy-data/time_series_covid19_confirmed_global.csv')
#recovered = pd.read_csv('data/src/pandemy-data/time_series_covid19_recovered_global.csv')
#death = pd.read_csv('data/src/pandemy-data/time_series_covid19_deaths_global.csv')
look_tbl = pd.read_csv('data/src/pandemy-data/UID_ISO_FIPS_LookUp_Table.csv')

look_tbl = look_tbl.drop_duplicates(subset=['Country_Region'], keep='first', ignore_index=True).set_index('Country_Region')

cases['iso_code'] = cases['Country/Region'].apply(lambda x : look_tbl.loc[x].iso3)
cases = cases.dropna(subset=['iso_code']) # rimuove elementi non riconducibili a paesi
cases = cases.drop(columns=['Province/State','Lat','Long', 'Country/Region'])
cases = cases.groupby('iso_code').sum() # raggruppa per iso_code

'''