import pandas as pd
import numpy as np
import pickle

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px 

from app import app

## Data loading
data = pd.read_pickle("data/data.pkl")


## Functions



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
            html.H4('Informazioni sugli utenti'),
            html.P(f"Sono presenti utenti unici."),
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

