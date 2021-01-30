import pandas as pd
import pickle

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px 


## App configs
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

## Data loading
data = pd.read_pickle("data/data.pkl")



## App layout
app.layout = html.Div(children=[
    html.H1(children='EDA'),
    html.H3(children='Prova'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure= px.imshow(data.isnull(), title="Valori nulli", labels=dict(x="Attributi")) 
    ),

    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.head(100).to_dict('records'),
        style_cell=dict(textAlign='left'),
        #style_header=dict(backgroundColor="paleturquoise"),
        #style_data=dict(backgroundColor="lavender"),
        style_table={
            'height': 500,
            'overflowY': 'auto',
            'width': 'auto',
        },
    )
])



## Callbacks



## Server startup
if __name__ == '__main__':
    app.run_server(debug=True)