import pandas as pd
import numpy as np
import pickle

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from app import app

## Data loading
data = pd.read_pickle("data/data.pkl")
vader_sent = pd.read_pickle("data/vader_sent.pkl")
roberta_sent = pd.read_pickle("data/roberta_sent.pkl")

# Sentiments comparison
tabledata = vader_sent.join(roberta_sent, how='outer', lsuffix='_vader', 
                            rsuffix='_roberta').drop(columns=['text-clean_vader', 'text_roberta','text-clean_roberta'])

# Sentiment analysis table
data_sent = data
data_sent['sentiment'] = vader_sent['sentiment']

## Functions
def sentBar(data1,data2):
    df1 = data1.sentiment.value_counts().reset_index()
    df1.columns=['SENTIMENT','COUNT']
    df2 = data2.sentiment.value_counts().reset_index()
    df2.columns=['SENTIMENT','COUNT']

    figBar = go.Figure()
    figBar.add_trace(go.Bar(
        x=df1['SENTIMENT'],
        y=df1['COUNT'],
        name='Vader',
        #marker_color='indianred'
    ))
    figBar.add_trace(go.Bar(
        x=df2['SENTIMENT'],
        y=df2['COUNT'],
        name='RoBERTa',
        #marker_color='lightsalmon'
    ))
    figBar.update_layout(barmode='group')
    figBar.update_traces(texttemplate='%{value}', textposition='inside')

    return figBar

def tweetTimeLineSent(data_sent):
    #df = data.date.dt.date.value_counts().reset_index()
    #df.columns=['DATE','COUNT']
    #df = df.sort_values('DATE')
    #figLine = px.line(df, x='DATE', y='COUNT', title='Numero di tweets nel tempo')

    #df2 = data.hashtags.dropna()
    #idx = df2.apply(lambda x: True if tag in x else False)
    #df2 = data.loc[idx.index[idx==True].tolist()]
    #df2 = df2.date.dt.date.value_counts().reset_index()
    #df2.columns = ['DATE','COUNT']
    #df2 = df2.sort_values('DATE')

    df_sent = pd.concat([data_sent, pd.get_dummies(data.sentiment)], axis=1)
    df_sent = df_sent.groupby(df_sent.date.dt.date).sum()
    #figLine = px.line(df_sent, x=df_sent.index, y=['Negative', 'Neutral','Positive'])


    figLine = go.Figure()
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Negative,
                        name='Negative',
                        line=dict(color='red')))
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Neutral,
                        name='Neutral',
                        line=dict(color='yellow')))
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Positive,
                        name='Positive',
                        line=dict(color='green')))


    return figLine


## Page layout
layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H1('Sentiment Analysis', className='text-white'),
            html.P('Visualizzazione e analisi del sentiment estratto.', className='mb-0 text-white')
        ]),
        className='py-3 mb-3 bg-dark shadow rounded',
    ),
    dbc.Tabs([
            dbc.Tab(label="Sentiment extraction", tab_id="tab-1-sent"),
            dbc.Tab(label="Sentiment analysis", tab_id="tab-2-sent"),
        ],
        id="tabs-sent",
        active_tab="tab-1-sent",
        className='shadow rounded-top'
    ),
    html.Div(id="tabs-sent-content"),
])


# Sentiment extraction
sent_extr_layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H4('Estrazione del sentiment'),
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in tabledata.columns],
                        data=tabledata.to_dict('records'),
                        style_cell={
                            'textAlign':'left',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        page_size=5,
                        sort_action="native",
                        sort_mode="single",
                    ),
                    className="p-1 m-5 shadow bg-white rounded"
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=sentBar(vader_sent,roberta_sent),
                        className='border shadow-sm rounded px-2'
                    ),
                ),
            ),
            
        ]),
        className='p-4 mb-3 bg-white rounded-bottom shadow'
    ),
])







# Sentiment analysis
sent_analy_layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H4('Informazioni sugli utenti'),
            html.P(f"Sono presenti utenti unici."),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=tweetTimeLineSent(data_sent),
                        className='border shadow-sm rounded'
                    )
                ),
                align='center',
                justify='center'
            ),




        ]),
        className='p-4 mb-3 bg-white rounded-bottom shadow'
    ),
])


## Callbacks
# Tab switcher
@app.callback(Output('tabs-sent-content', 'children'),
              [Input('tabs-sent', 'active_tab')])
def render_content(tab):
    if tab == 'tab-1-sent':
        return sent_extr_layout
    elif tab == 'tab-2-sent':
        return sent_analy_layout

