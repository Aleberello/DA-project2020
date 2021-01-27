import pandas as pd
import string
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
#nltk.download('punkt')

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def preproc(text):
    new_text = re.sub(r'http\S+', '', text) #urls
    new_text = re.sub(r'@[^\s]+', '', new_text) #mentions
    #new_text = new_text.lower() #lowercase
    new_text = re.sub(r'—|’|’’|-|”|“|‘', ' ', new_text) #separators and quotes
    new_text = new_text.strip()
    new_text = re.sub(r'\d+', '', new_text) #digits
    new_text = new_text.translate(str.maketrans('', '', string.punctuation)) #punteggiatura
    new_text = deEmojify(new_text)
    new_text = WordPunctTokenizer().tokenize(new_text)
    return new_text


def get_country(row):
    tmp = []
    #import pdb; pdb.set_trace()
    countries,cities = get_codes()

    import pdb; pdb.set_trace()
    if countries.isin(row).any().any():
        yep = [i for i, x in enumerate(countries.isin(row).any()) if x][0]
        tmp = countries[countries.iloc[:,yep]==token]['code3'].values[0]



    for idx, token in enumerate(row):
        # Se trova il codice a 2 o 3 char, o il country name salta gli altri token
        # if token in countries['code2'].tolist():
        #     tmp = (countries[countries['code2']==token]['code3'].values[0])
        #     break
        # elif token in countries['code3'].tolist():
        #     tmp = token
        #     break
        # elif token in countries['country'].tolist():
        #     tmp = token
        #     break
        # Cerca anche nei nomi alternativi dei paesi
        #if False:
        #    print(x)
        #else:
        #    if countries.isin([token]).any().any():
        #        yep = [i for i, x in enumerate(countries.isin([token]).any()) if x][0]
        #        tmp = countries[countries.iloc[:,yep]==token]['code3'].values[0]
        #        break
                #for idx,val in enumerate(countries.isin([token]).any()):
                #    if val:
                #        import pdb; pdb.set_trace()
                #        tmp = countries[countries.iloc[:,idx]==token]['code3'].values[0]
                #        break


        # Se trova il nome di una città, aggiungi per la valutazione
        if token in cities['name'].values:
            #import pdb; pdb.set_trace()
            tmp_idx = cities[cities['name']==token]['country'].index.values
            #import pdb; pdb.set_trace()
            if len(tmp_idx) > 1 and idx+1<len(row): # trovate città omonime, guardo il token successivo
                if row[idx+1] in cities.iloc[tmp_idx]['country'].values:
                    tmp2 = cities.iloc[tmp_idx][cities.iloc[tmp_idx]['country']==row[idx+1]]
                    tmp = tmp2['country'].values[0]
                    break
                elif row[idx+1] in cities.iloc[tmp_idx]['subcountry'].values:
                    tmp2 = cities.iloc[tmp_idx][cities.iloc[tmp_idx]['subcountry']==row[idx+1]]
                    tmp = tmp2['country'].values[0]
                    break
            else:
                tmp_country = cities.iloc[tmp_idx]['country'].values[0]
                if countries.isin([tmp_country]).any().any():
                    yep = [i for i, x in enumerate(countries.isin([tmp_country]).any()) if x][0]
                    import pdb; pdb.set_trace()
                    tmp = countries[countries.iloc[:,yep]==tmp_country]['code3'].values[0]
                    break


    if len(tmp) <1:
        tmp = None

    print(row, '->', str(tmp))
    return tmp



def get_codes():
    cities = pd.read_csv('./utils/world-cities.csv')
    countries_raw = pd.read_csv('./utils/country_iso_codes_expanded.csv')
    countries = countries_raw.drop(countries_raw.columns[4],axis=1)
    #countries = countries_raw.drop(countries_raw.columns[3:],axis=1)
    #countries['alternatives'] = countries_raw.iloc[:,5:].values.tolist()
    # remove nan
    #countries['alternatives'] = countries.alternatives.apply(lambda row: [x for x in row if str(x) != 'nan'])
    countries.fillna('None')
    countries.rename(columns={'Alpha-2 code': 'code2', 'Alpha-3 code': 'code3'}, inplace=True)

    return countries, cities



def get_countries(df):

    # Dizionario
    #countries,cities = get_codes()
    #import pdb; pdb.set_trace()



    df = df.fillna('None')
    good = df.apply(preproc).apply(get_country)

    #df.iloc[14:20][0].apply(preproc).apply(get_country)
    
    #good = df[0].apply(preproc).apply(get_country)
