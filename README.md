# Sentiment analysis COVID19 vaccine tweets
## Progetto Data Analytics, a.a. 19-20
Alessandro Bertolo (808314) - a.bertolo2@campus.unimib.it

## Struttura del repository

```bash
├── apps/                           # contiene le pagine principali che compongono la dashboard delle visualizzazioni
│   ├── eda.py                      # pagina per l'EDA
│   ├── home.py                     # pagina di homepage
│   └── sentiment.py                # pagina del sentiment extraction e analysis
├── data/                           # contiene i dataframe ottenuti
│   ├── src/                        # contiene i csv dei dataset di partenza
│       ├── pandemy-data/           # contiene i csv sull'andamento dell'epidemia e delle campagne vaccinali 
│       │   └── ...
│       ├── covidvaccine.csv
│       └── vaccination_tweets.csv
│   ├── data_tokens.pkl             # dataframe contenente i token estratti dal tokenizer
│   ├── data.pkl                    # dataframe del dataset ottenuto dopo il preprocessing
│   ├── roberta_sent.pkl            # dataframe con il sentiment estratto tramite RoBERTa
│   └── vader_sent.pkl              # dataframe con il sentiment estratto tramite Vader
├── utils                           # utils folder
│   ├── geo-data                    # contiene i dataset utilizzati per la standardizzazione del campo user_location 
│       └── ...
│   ├── sentiment_extraction.py     # contiene i metodi utilizzati per estrarre il sentiment
│   └── utils.py                    # contiene i metodi di tokenizatione e di standardizzazione del campo user_location
├── app.py                          # script di configurazione della dashboard per le visualizzazioni
├── indexer.py                      # script di avvio della dashboard per le visualizzazioni
├── main.py                         # contiene tutte le fasi di processing eseguite sul dataset
├── README.md
├── requiments.txt
└── .gitignore
```


## Componenti principali
### Processing dei dati
L'elaborazione è finalizzata all'ottenimento di dataframe contenenti tutte le informazioni necessarie per la fase di visualizzazione e di analisi.
Il file `main.py` contiene tutte le operazioni effettuate sui dati gerzzi, in particolare:
1. Caricamento dei dati -> merging dei due dataset
2. Data cleaning -> gestione tipi di dato e valori nulli, standardizazione di attributi, estrazione tokens da campo text
3. Sentiment extraction -> estrazione del sentiment utilizzando due tecniche rule-based (Vader e Roberta tramite la libreria `transformers`)
4. Caricamento dati ausiliari -> dataset riferiti all'andamento dell'epidemia e della campagna vaccinale mondiale

I dataframe ottenuti vengono salvati nella cartella `"data/"` come oggetti *pickle* per alleggerire il carico della dashboard per le visualizzazioni.

### Visualizzazione ed analisi
Per la parte di visualizzazione ed analisi è stata implementata una dashboard multipagina utilizzando la piattaforma Dash di Plotly. 
La progettazione delle pagine è stata effettuata utilizzando il Bootstrap Component, mentre i grafici mediante le librerie messe a disposizione da Plotly. Il file `index.py` è il punto di avvio per la dashboard. Le pagine che la compongono sono contenute nella cartella `"apps/"`.


## Come avviare
### Requiments
Le librerie richieste per il progetto sono contenute nel file `requiments.txt`:
```
pip install requiments.txt
```
Per le parti di preprocessing ed estrazione del sentiment è necessario effettuare il download di alcuni package aggiuntivi:
- per NLTK, decommentando le seguenti righe di codice nello script `"utils/utils.py"`
    ```python
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```
- per il sentiment extraction, il modello addestrato di RoBERTa viene scaricato alla prima esecuzione del codice

### Avvio dashboard
La dash contenente le visualizzazioni va avviata eseguendo lo script `index.py`. Una volta inizializzato il server Flask, la pagina è visibile nel localhost http://127.0.0.1:8050/.


## Note

## Citazioni
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
- Rosenthal, Sara, Noura Farra, and Preslav Nakov. "SemEval-2017 task 4: Sentiment analysis in Twitter." Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017). 2017.
- Bird, Steven, Edward Loper and Ewan Klein (2009).
Natural Language Processing with Python.  O'Reilly Media Inc.
- Wolf, Thomas, et al. "Transformers: State-of-the-art natural language processing." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 2020.
- Dong, Ensheng, Hongru Du, and Lauren Gardner. "An interactive web-based dashboard to track COVID-19 in real time." The Lancet infectious diseases 20.5 (2020): 533-534.