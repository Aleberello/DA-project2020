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

layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H1('Home', className='text-white'),
        ]),
        className='py-3 mb-3 bg-dark shadow rounded',
    ),
    dbc.Row(
        dbc.Col([
            html.H4('Introduzione'),
            dcc.Markdown('''
                Dall’inizio della pandemia di Covid19, uno degli argomenti maggiormente trattato è quello dei vaccini. Social Networks come Twitter, in questi mesi, hanno subito un’accelerazione nella quantità di informazioni prodotte dagli utenti riguardanti la malattia e tutto ciò che la circonda. Risulta quindi interessante analizzare ciò che gli utenti scrivono a riguardo.  
                Uno degli argomenti di maggior divisione sociale riguarda lo sviluppo del vaccino e la campagna vaccinale.
                Tramite l’analisi di questi tweet il progetto mira a rispondere ad alcune domande:  
                - è possibile identificare una tendenza preponderante nel sentiment dei tweets riferiti ai vaccini?
                - in quali paesi l’argomento dei vaccini è più presente? è possibile identificare una polarità preponderante per ciascun paese?
                - tra gli utenti con maggiore seguito, vi è una tendenza maggiore al sentiment positivo o negativo?  
                  
                Non avendo a disposizione dei dati già etichettati con il sentiment, ai fini del progetto è necessario classificare i tweets raccolti. Per farlo vengono utilizzati due metodologie. Le domande correlate sono quindi:
                - dai due approcci di sentiment extraction si ottengono risultati simili?
                - si riesce ad ottenere una buona astrazione? 
                - quale dei due è migliore?
            '''),
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id='data-table',
                        columns=[{"name": i, "id": i} for i in data.columns],
                        data=data.to_dict('records'),
                        style_table={
                            'overflowX' : 'auto',
                            'width': '100%',
                        },
                        style_cell={
                            'textAlign':'left',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'lineHeight': '15px',
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        page_size=5,
                        sort_action="native",
                        sort_mode="single",
                    ),
                    className="p-1 shadow bg-white rounded"
                )
            ),
        ]),
        className='p-4 mb-3 bg-white rounded-bottom shadow'
    ),
])
