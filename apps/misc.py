
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



## Page layout
layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            dcc.Graph(
                id='null-heatmap',
                figure= px.imshow(data.isnull(), title="Valori mancanti nel dataset", labels=dict(x="Attributi")) 
            ),
            className="p-1 m-5 shadow bg-white rounded"
        )
    ),

    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in data.columns],
                data=data.head(100).to_dict('records'),
                style_cell=dict(textAlign='left'),
                style_table={
                    'height': 500,
                    'overflowY': 'auto',
                    'width': 'auto',
                }
            ),
            className="p-1 m-5 shadow bg-white rounded"
        )
    )
])
