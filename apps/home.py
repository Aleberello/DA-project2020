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
            html.H4('Introduzione'),
            dcc.Markdown('''
                Descrizione
            #'''),
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
