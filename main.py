### Libraries import
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from utils.utils import *


### Caricamento dati
pfizer = pd.read_csv('datasets/vaccination_tweets.csv', parse_dates=[4,9], infer_datetime_format=True)
vacc = pd.read_csv('datasets/covidvaccine.csv', parse_dates=[3,8], infer_datetime_format=True)

# Join dei tweets
comb = pd.concat([pfizer[vacc.columns], vacc], ignore_index=True)
# Rimozione duplicati basandosi su text, user_name e date
data = comb.drop_duplicates(subset=['user_name','date','text'], keep='first', ignore_index=True)

### Data engineering
# Standardizzazione locations
data['Country'] = get_countries(data.user_location)

### EDA



import pdb; pdb.set_trace()