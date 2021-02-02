import numpy as np
import pandas as pd
import csv
import urllib.request
import re
from tqdm import tqdm

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress tf cuda warning
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax



def get_vader_sentiment(texts):
    '''

    '''

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
        Estrae la label del sentiment dallo score compound estratto dall'analyzer Vader, le soglie sono quelle
        consigliate dagli autori del metodo.
        '''
        if score>=0.05:
            return 'Positive'
        elif score<=(-0.05):
            return 'Negative'
        else:
            return 'Neutral'

    ## Setup dell'analyzer
    vader_analyzer = SentimentIntensityAnalyzer()

    ## Sentiment extraction
    texts['text-clean'] = texts['text'].apply(lambda x : vader_preproc(x))
    texts['score'] = texts['text-clean'].apply(lambda x : vader_analyzer.polarity_scores(x)['compound'])
    texts['sentiment'] = texts['score'].apply(get_vader_sent)

    return texts




def get_roberta_sentiment(texts):
    '''

    '''

    def download_labels():
        '''
        Funzione ausiliaria per il mapping delle label restituite dal modello, utilizzato solo come riferimento.
        '''
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
        return labels

    def rob_preproc(text):
        '''
        Funzione di preprocessing definita dagli autori ed utilizzata anche per il training del modello.
        Sotituisce le menzioni e i link con delle keyword.
        Aggiunta solamente la rimozione degli escape.
        '''
        new_text = []
        text = re.sub(r'[\n\r\t]', '', text) #char escapes
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)


    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    labels = {'LABEL_0':'Negative','LABEL_1':'Neutral','LABEL_2':'Positive'}

    ## Model loading e setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    pipel = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, return_all_scores=False)
    print('Model loaded.')

    ## Sentiment extraction
    tqdm.pandas()
    texts['text-clean'] = texts['text'].progress_apply(rob_preproc)
    sentiments = texts['text-clean'].progress_apply(pipel)

    senti_labels = []
    senti_scores = []
    for row in sentiments:
        senti_labels.append(labels[row[0]['label']])
        senti_scores.append(np.round(row[0]['score'], 4))

    texts['score'] = senti_scores
    texts['sentiment'] = senti_labels

    return texts