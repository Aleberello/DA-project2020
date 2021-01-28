### Libraries import
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import pickle
import plotly.express as px 

from utils.utils import *



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

import pdb; pdb.set_trace()



#### Sentiment Analysis