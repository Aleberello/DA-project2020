
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

num_usr = len(data.user_name.unique())


## Functions
def sourcePie(data):
    sourcedf = data.source.value_counts().reset_index()
    sourcedf.columns=['SOURCE','COUNT']
    sourcedf.loc[sourcedf['COUNT'] < 1000, 'SOURCE'] = 'Other sources'

    figPie = px.pie(sourcedf, values='COUNT', names='SOURCE', title='Principali software utilizzati per i tweets')

    figPie.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return figPie


def uniqueUsersBar(data):
    usersdf = data.user_name.value_counts().reset_index()
    usersdf.columns=['USER','TWEET_COUNT']
    usersdf = usersdf.iloc[0:10]
    #others = usersdf[usersdf['TWEET_COUNT'] < 100]
    #usersdf = usersdf.drop(others.index)
    #usersdf = usersdf.append({'USER':'Others', 'TWEET_COUNT':others['TWEET_COUNT'].sum()}, ignore_index=True)
    #usersdf['PERC'] = np.around(((usersdf['TWEET_COUNT'] / len(data))*100),decimals=2)

    figBar = px.bar(usersdf,
                    x='TWEET_COUNT', y="USER", 
                    orientation='h',
                    text='TWEET_COUNT', title="Top-10 tweeters")
    figBar.update_traces(texttemplate='%{value}', textposition='inside')
    figBar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    figBar.update_layout({
        #'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return figBar


def locationsMapPlot(data):
    wmap = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
    countrydf = data.country.value_counts().reset_index()
    countrydf.columns=['CODE','TWEET_COUNT']
    wmap_count = pd.merge(wmap, countrydf, how='left', on='CODE')
    wmap_count = wmap_count.fillna(0)

    figMap = px.choropleth(wmap_count, locations="CODE",color="TWEET_COUNT", hover_name="COUNTRY",
                            color_continuous_scale=px.colors.sequential.Plasma)
    return figMap





## Page layout
layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H1('EDA', className='text-white'),
            html.P('Grafici di Explorative Data Analysis eseguiti sul dataset.', className='mb-0 text-white')
        ]),
        className='py-3 mb-3 bg-dark shadow rounded',
    ),
    
    dbc.Row(
        dbc.Col([
            html.H4('Info sugli utenti'),
            html.P(f"Sono presenti {num_usr} utenti unici."),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        figure=sourcePie(data),
                        className='border shadow-sm rounded px-2'
                    ),
                    align='center',
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=uniqueUsersBar(data),
                        className='border shadow-sm rounded px-2'
                    ),
                    align='center',
                    width=6
                )],
                align='center',
                justify='center',
                className='mb-3'
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=locationsMapPlot(data),
                        className='border shadow-sm rounded'
                    )
                ),
                align='center',
                justify='center'
            )
        ]),
        className='p-4 mb-3 bg-white rounded shadow'
    ),
])
