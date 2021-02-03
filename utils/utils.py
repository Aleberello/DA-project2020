import pandas as pd
import string
import re
import emoji
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')


def get_tokens(text):
    '''
    Tokenizer per il campo text dei tweets estratti, utilizzati nelle visualizzazioni.
    '''
    lemmatizer = WordNetLemmatizer()

    stops = stopwords.words('english')
    # Dopo aver analizzato le parole più frequenti vengono aggiunti alcuni termini da scartare
    stops =set(stops)
    stops.add("amp")
    stops.add("u")
    stops.add("\u2026")
    stops = list(stops)

    new_text = text.lower()
    new_text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', new_text) #urls
    new_text = re.sub(r'(?:@[\w_]+)', '', new_text) #mentions
    new_text = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', new_text) #hash-tags
    new_text = re.sub(r'—|’|’’|-|”|“|‘', ' ', new_text) #separators and quotes
    new_text = new_text.strip()
    new_text = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', new_text) #numbers
    new_text = new_text.translate(str.maketrans('', '', string.punctuation)) #punteggiatura
    new_text = re.sub(r'<[^>]+>', ' ', new_text) # HTML tags
    new_text = re.sub(emoji.get_emoji_regexp(), ' ', new_text).strip() #emoji removal
    new_text = WordPunctTokenizer().tokenize(new_text)
    new_text = [item for item in new_text if item not in stops] #stop-words
    new_text = [lemmatizer.lemmatize(word) for word in new_text]
    return new_text



def get_countries(df):
    '''
    Processa la colonna locations e restituisce la standardizzazione (3-char code)
    '''

    def get_codes():
        '''
        Carica i dizionari utilizzati per la standardizzazione della locations
        '''
        cities = pd.read_csv('./utils/geo-data/world-cities.csv')
        usa_states = pd.read_csv('./utils/geo-data/usa-countries-code.csv')
        countries_raw = pd.read_csv('./utils/geo-data/country_iso_codes_expanded.csv')
        countries = countries_raw.drop(countries_raw.columns[4],axis=1)
        countries.fillna('None')
        countries.rename(columns={'Alpha-2 code': 'code2', 'Alpha-3 code': 'code3'}, inplace=True)
        return countries, cities, usa_states

    def loc_preprocess(text):
        '''
        Custom tokenizer per il campo location
        '''           
        new_text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', text) #urls
        new_text = re.sub(r'(?:@[\w_]+)', '', new_text) #mentions
        new_text = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', new_text) #hash-tags
        new_text = re.sub(r'—|’|’’|-|”|“|‘', ' ', new_text) #separators and quotes
        new_text = new_text.strip()
        new_text = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', new_text) #numbers
        new_text = new_text.translate(str.maketrans('', '', string.punctuation)) #punteggiatura
        new_text = re.sub(emoji.get_emoji_regexp(), ' ', new_text).strip() #emoji removal
        new_text = WordPunctTokenizer().tokenize(new_text)
        return new_text

    def get_country(row, countries, cities, usa):
        '''
        Per ciascuna location pre-processata, effettua la standardizzazione (3-char)
        '''
        tmp = []
        if not row or row[0]=='None':
            return None

        for idx, token in enumerate(row):
            # Se trova il codice a 2 o 3 char, il nome o i nomi alternativi di uno stato del mondo
            if token in countries['code2'].tolist():
                tmp = (countries[countries['code2']==token]['code3'].values[0])
                break
            elif token in countries['code3'].tolist():
                tmp = token
                break
            elif token in countries['country'].tolist():
                tmp = countries[countries['country']==token]['code3'].values[0]
                break
            elif countries.iloc[:,4:].isin([token]).any().any():
                column = [i for i, x in enumerate(countries.iloc[:,4:].isin([token]).any()) if x][0]
                tmp = countries[countries.iloc[:, column+4]==token]['code3'].values[0]
                break

            # Se trova il nome di una città
            if len(token) > 4 and token in cities['name'].values:
                tmp_idx = cities[cities['name']==token]['country'].index.values
                tmp_country = []
                if len(tmp_idx) > 1 and idx+1<len(row): # trovate città omonime
                    # guarda il token successivo
                    if row[idx+1] in cities.iloc[tmp_idx]['country'].values:
                        tmp2 = cities.iloc[tmp_idx][cities.iloc[tmp_idx]['country']==row[idx+1]]
                        tmp_country = tmp2['country'].values[0]
                    elif row[idx+1] in cities.iloc[tmp_idx]['subcountry'].values:
                        tmp2 = cities.iloc[tmp_idx][cities.iloc[tmp_idx]['subcountry']==row[idx+1]]
                        tmp_country = tmp2['country'].values[0]
                else: # trovata una sola città
                    tmp_country = cities.iloc[tmp_idx]['country'].values[0]

                # estrae il codice dalla città trovata
                if countries.isin([tmp_country]).any().any():
                    column = [i for i, x in enumerate(countries.isin([tmp_country]).any()) if x][0]
                    tmp = countries[countries.iloc[:,column]==tmp_country]['code3'].values[0]
                    break

            # Se trova il nome di uno stato USA
            if token in usa['State'].values or token in usa['Abbreviation'].values:
                tmp = 'USA'
        
        # Nessuna corrispondenza geografica trovata
        if len(tmp)<1:
            tmp = None

        print(row, '->', str(tmp))
        return tmp


    # Caricamento dizionario dei paesi, città e stati usa
    countries, cities, usa_states = get_codes()
    countries_codes = df.apply(loc_preprocess).apply(get_country, args=(countries,cities,usa_states))
    return countries_codes