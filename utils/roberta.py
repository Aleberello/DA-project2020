import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import pandas as pd
import csv
import urllib.request


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
    Funzione di preprocessing definita dagli autori
    '''
    new_text = []
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    pipel = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, return_all_scores=False)

    import pdb; pdb.set_trace()

    dftext = pd.DataFrame(texts[0:100])
    dftext['prep_text'] = dftext.text.apply(rob_preproc)

    sentiments = dftext.prep_text.apply(pipel)

    senti_labels = []
    senti_scores = []
    for row in sentiments:
        senti_labels.append(labels[row[0]['label']])
        senti_scores.append(np.round(row[0]['score'], 4))

    dftext['sentiment'] = senti_labels
    dftext['score'] = senti_scores

    return dftext

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