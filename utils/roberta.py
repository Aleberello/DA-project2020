import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress tf cuda warning
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import numpy as np
import pandas as pd
import csv
import urllib.request
import re
from tqdm import tqdm


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


def get_roberta_sentiment(texts):
    '''

    '''
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    labels = {'LABEL_0':'Negative','LABEL_1':'Neutral','LABEL_2':'Positive'}

    ## Model loading and setup
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

    #text = "Good night 😊"
    #text = rob_preproc(text)
    #encoded_input = tokenizer(text, return_tensors='pt')
    #output = model(**encoded_input) 
    #scores = output[0][0].detach().numpy()
    #scores = softmax(scores)

    #ranking = np.argsort(scores)
    #ranking = ranking[::-1]
    #for i in range(scores.shape[0]):
    #    l = labels[ranking[i]]
    #    s = scores[ranking[i]]
    #    print(f"{i+1}) {l} {np.round(float(s), 4)}")