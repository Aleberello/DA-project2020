import pandas as pd
import numpy as np
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


## Plot functions
def locationsMapPlot(data):
    wmap = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
    countrydf = data.country.value_counts().reset_index()
    countrydf.columns=['CODE','TWEET_COUNT']
    wmap_count = pd.merge(wmap, countrydf, how='left', on='CODE')
    wmap_count = wmap_count.fillna(0)

    figMap = px.choropleth(wmap_count, locations="CODE",color="TWEET_COUNT", hover_name="COUNTRY",
                            color_continuous_scale=px.colors.sequential.Plasma)
    return figMap

def locationsBarPlot(data):
    wmap = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
    countrydf = data.country.value_counts().reset_index()
    countrydf.columns=['CODE','TWEET_COUNT']
    wmap_count_bar = pd.merge(wmap, countrydf, how='left', on='CODE')
    wmap_count_bar = wmap_count_bar.fillna(0)
    wmap_count_bar = wmap_count_bar.sort_values('TWEET_COUNT', ascending=False, ignore_index=True).head(10)
    wmap_count_bar = wmap_count_bar.append({'COUNTRY':'Unknown', 'TWEET_COUNT':data.country.isnull().sum()}, 
                                            ignore_index=True)
    wmap_count_bar['PERC'] = np.around(((wmap_count_bar.TWEET_COUNT / len(data))*100),decimals=2)

    mark_colors = ['lightslategray'] * len(wmap_count_bar)
    mark_colors[-1] = 'crimson'

    figBar = px.bar(wmap_count_bar, x="TWEET_COUNT", y="COUNTRY", 
                                    orientation='h',
                                    text='PERC', title="Top tweets countries")
    figBar.update_traces(texttemplate='%{value} (%{text}%)', textposition='outside', marker_color=mark_colors)
    figBar.update_layout(yaxis={'categoryorder':'total ascending'})
    return figBar

def sourcePie(data):
    sourcedf = data.source.value_counts().reset_index()
    sourcedf.columns=['SOURCE','COUNT']
    sourcedf.loc[sourcedf['COUNT'] < 1000, 'SOURCE'] = 'Other sources'

    figPie = px.pie(sourcedf, values='COUNT', names='SOURCE')
    return figPie


## App layout
app.layout = html.Div(children=[
    html.H1(children='EDA'),
    html.H3(children='Prova'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='null-heatmap',
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
    ),

    html.Div([
        html.Div([
            html.H3('Column 1'),
            dcc.Graph(
                id='country-map',
                figure=locationsMapPlot(data)
            )],
            className="six columns"
        ),
        html.Div([
            html.H3('Column 1'),
            dcc.Graph(
                id='country-bar',
                figure=locationsBarPlot(data)
            )],
            className="six columns"
        )],
        className="row"
    ),

    dcc.Graph(
        id='source-pie',
        figure=sourcePie(data)
    )

])



## Callbacks



## Server startup
if __name__ == '__main__':
    app.run_server(debug=True)