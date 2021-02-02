# Sentiment exploration COVID19 vaccine tweets
## Progetto corso Data Analytics a.a. 19-20
Alessandro Bertolo (808314) - a.bertolo2@campus.unimib.it

## Repo structure

```bash
├── apps                        # extracted tweets for users and news
│   ├── eda.py
│   ├── home.py
│   ├── misc.py
│   └── sentiment.py
├── data                  # Twitter API implementation for tweets extraction
│   ├── src                         # Twitter API credentials
│       ├── covidvaccine.csv
│       └── vaccination_tweets.csv
│   ├── data.pkl                   # tweets scraper
│   ├── roberta_sent.pkl                   # tweets scraper
│   └── vader_sent.pkl               # list of Twitter usernames from wich extract tweets
├── utils                           # utils folder
│   ├── world-datasets               # stores all already user tweets pre-processed for personalization in pickle files
│       └── ...
│   ├── utils.py                    # utils variables and methods
│   └── roberta.py                  # WordNet synonyms dictionary used for synonyms queries in ElasticSearch
├── demo.py                         # demo script for the project
├── indexer.py                      # script used for indexing tweets in ElasticSearch
├── preprocessor.py                 # script used for manual pre-processing of tweets and query personalization phase
├── README.md
├── requiments.txt
└── .gitignore
```

## Descrizione
Il progetto mira a....
Due dataset accorpati...


### Componenti principali
#### Elaborazione dei dati
L'elaborazione è finalizzata all'ottenimento di dataframe contenenti tutte le informazioni necessarie per la fase di visualizzazione e di analisi.
Il file `main.py` contiene tutte le operazioni effettuate sui dati gerzzi, in particolare:
1. Caricamento dei dati -> merging dei due dataset
2. Data cleaning -> gestione tipi di dato e valori nulli, standardizazione di attributi
3. Sentiment extraction -> estrazione del sentiment utilizzando due tecniche rule-based (Vader e Roberta tramite la libreria `transformers`)
4. Caricamento dati correlati -> dataset riferiti all'andamento dell'epidemia e della campagna vaccinale mondiale

I dataframe ottenuti vengono salvati nella cartella `"data/"` come oggetti *pickle*.

#### Visualizzazione ed analisi
Per la parte di visualizzazione ed analisi è stata implementata una Dashboard multipagina. Il file `index.py` è il punto di avvio per la dashboard. Le pagine che la compongono sono contenute nella cartella `"apps/"`.


## Come avviare
### Requiments
Le librerie richieste per il progetto sono contenute nel file `requiments.txt`:
```
pip install requiments.txt
```

### Set-up
- Download and install [Elasticsearch 7.10.1](https://www.elastic.co/downloads/elasticsearch)
- For the synonym queries:
    - move WordNet dictionary from `"utils/wn_s.pl"` to ElasticSearch source folder `"elasticsearch-*/config/"`
- For the preprocessing section (`processor.py`):
    - un-comment and download, only for the first execution, the following nltk packages
    ```python
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```



## Note
- After the first execution the preprocessed tweets of given JSON are saved into *JSON_filename.pickle* file 
in `"./utils/user-profiles"` optimize the execution time
- After the first execution TfidfVectorized object for each user are saved into a .pickle file in 
`"./utils/user-profile/*user_name*"` folder to optimize the execution time


## Citations