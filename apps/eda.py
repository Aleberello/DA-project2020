import pandas as pd
import numpy as np
import pickle

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
from wordcloud import WordCloud

from app import app


## Data loading
data = pd.read_pickle("data/data.pkl")

num_usr = len(data.user_name.unique())
num_tweets = len(data.text.unique())
hashtags = data.hashtags.dropna().explode().value_counts().head(50).index.tolist() #top-50 hashtags list
num_words = round(data.text.apply(lambda x: len(x.split())).mean())


## Functions
# Users functions
def verifiedPie(data):
    df = data.drop_duplicates(subset=['user_name'], keep='first', ignore_index=True).user_verified.value_counts().reset_index()
    df.columns=['STATUS','COUNT']
    df.loc[df['STATUS']==True, ['STATUS']] = 'Verified'
    df.loc[df['STATUS']==False, ['STATUS']] = 'Unverified'

    figPie = px.pie(df, values='COUNT', names='STATUS', title='Utenti verificati',
                    color='STATUS',
                    color_discrete_map={'Verified':'#636EFA', 'Unverified':'#EF553B'})
    return figPie

def followersBar(data, num=10):
    df = data[['user_name', 'user_followers', 'user_verified']]
    df.columns=['USER','FOLLOWERS','STATUS']
    df = df.groupby('USER').max().sort_values('FOLLOWERS', ascending=False).head(num)
    df.loc[df['STATUS']==True, ['STATUS']] = 'Verified'
    df.loc[df['STATUS']==False, ['STATUS']] = 'Unverified'

    figBar = px.bar(df,
                    x='FOLLOWERS', y=df.index, 
                    orientation='h', color='STATUS',
                    text='FOLLOWERS', title=f"Top-{num} account seguiti per numero di followers",
                    color_discrete_map={'Verified':'#636EFA', 'Unverified':'#EF553B'})
    figBar.update_traces(texttemplate='%{value}', textposition='inside')
    figBar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=True)
    return figBar

def uniqueUsersBar(data, num=10):
    df = data[['user_name','user_verified']].value_counts().reset_index()
    df.columns=['USER','STATUS','TWEETS']
    df.loc[df['STATUS']==True, ['STATUS']] = 'Verified'
    df.loc[df['STATUS']==False, ['STATUS']] = 'Unverified'

    df = df.iloc[0:num]

    figBar = px.bar(df,
                    x='TWEETS', y="USER", 
                    orientation='h', color='STATUS',
                    text='TWEETS', title=f"Top-{num} account per numero di tweets",
                    color_discrete_map={'Verified':'#636EFA', 'Unverified':'#EF553B'})
    figBar.update_traces(texttemplate='%{value}', textposition='inside')
    figBar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=True)
    return figBar

def locationsMapPlot(data):
    wmap = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
    countrydf = data.country.value_counts().reset_index()
    countrydf.columns=['CODE','TWEETS']
    wmap_count = pd.merge(wmap, countrydf, how='left', on='CODE')
    wmap_count = wmap_count.fillna(0)

    figMap = px.choropleth(wmap_count, locations="CODE",color="TWEETS", hover_name="COUNTRY",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title='Nazioni di provenienza degli utenti per numero di tweets')
    figMap.update_coloraxes(showscale=False)
    return figMap

def locationsBar(data, num):
    df = data.country.value_counts().reset_index()
    df.columns=['CODE','TWEETS']
    df = df.iloc[0:num]

    figBar = px.bar(df,
                    x='TWEETS', y="CODE", 
                    orientation='h', color='TWEETS',
                    text='TWEETS', title=f"Top-{num} nazioni per numero di tweets",
                    color_continuous_scale=px.colors.sequential.Plasma)
    figBar.update_traces(texttemplate='%{value}', textposition='inside')
    figBar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=True)
    return figBar


# Tweets functions
def sourcePie(data):
    sourcedf = data.source.value_counts().reset_index()
    sourcedf.columns=['SOURCE','COUNT']
    sourcedf.loc[sourcedf['COUNT'] < 1000, 'SOURCE'] = 'Other sources'

    figPie = px.pie(sourcedf, values='COUNT', names='SOURCE', title='Principali fonti di provenienza dei tweets')
    return figPie


def tweetTimeLine(data, tag):
    df = data.date.dt.date.value_counts().reset_index()
    df.columns=['DATE','COUNT']
    df = df.sort_values('DATE')
    figLine = px.line(df, x='DATE', y='COUNT', title='Numero di tweets nel tempo')

    df2 = data.hashtags.dropna()
    idx = df2.apply(lambda x: True if tag in x else False)
    df2 = data.loc[idx.index[idx==True].tolist()]
    df2 = df2.date.dt.date.value_counts().reset_index()
    df2.columns = ['DATE','COUNT']
    df2 = df2.sort_values('DATE')

    figLine.add_scatter(x=df2.DATE, y=df2.COUNT, mode='lines', name=tag)
    figLine.update_traces(mode="markers+lines")
    figLine.update_layout(hovermode="x unified")
    return figLine


def hashtagsBar(data):
    df = data.hashtags
    zero = len(df)-df.isnull().sum()
    df = df.dropna()
    df = df.apply(lambda x: len(x)).value_counts().reset_index()
    df.columns=['NUM_HASHTAGS','COUNT']
    df = df.append({'NUM_HASHTAGS':0, 'COUNT':zero}, ignore_index=True)
    mean = round(df['COUNT'].mean())

    figBar = px.bar(df, x='NUM_HASHTAGS', y='COUNT', title="Numero di hashtags per tweet")
    figBar.update_traces(texttemplate='%{value}', textposition='outside')
    figBar.update_layout(xaxis={'tickmode':'linear'})
    figBar.add_hline(y=mean, line_dash="dash", line_color="red",
                    annotation_text=f"Media: {mean}", 
                    annotation_position="top right")
    figBar.add_vline(x=2, line_dash="dash", line_color="red",
                    annotation_text=f"Numero medio di hashtags utilizzati", 
                    annotation_position="top right")
    return figBar


def topHashtagsBar(data):
    df = data.hashtags.dropna()
    df = df.explode().value_counts().reset_index().head(10)
    df.columns=['HASHTAG','COUNT']

    figBar = px.bar(df, x='COUNT', y='HASHTAG', 
                    orientation='h',
                    text='COUNT', title="Top-10 hashtags utilizzati")
    figBar.update_traces(texttemplate='%{value}', textposition='inside')
    figBar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    return figBar


def textWordcloud(data):
    df = data.hashtags.dropna()
    df = df.explode().value_counts()
    #words_freq = data.text.str.split(expand=True).stack().value_counts()
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(df)
    fig = px.imshow(wc)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(hovermode=False)
    return fig



## Page layout
layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H1('EDA', className='text-white'),
            html.P('Explorative Data Analysis divisa tra utenti e tweets.', className='mb-0 text-white')
        ]),
        className='py-3 mb-3 bg-dark shadow rounded',
    ),
    dbc.Tabs([
            dbc.Tab(label="Utenti", tab_id="tab-1-eda"),
            dbc.Tab(label="Tweets", tab_id="tab-2-eda"),
        ],
        id="tabs-eda",
        active_tab="tab-1-eda",
        className='shadow rounded-top'
    ),
    html.Div(id="tabs-eda-content"),
])


# Users eda
users_layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H4('Informazioni sugli utenti'),
            html.P(f"Nel dataset sono presenti {num_usr} utenti unici."),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        figure=verifiedPie(data),
                        className='border shadow-sm rounded px-2'
                    ),
                    align='center',
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(
                        id='slider-unique-bar',
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
                dbc.Col([
                    dcc.Graph(
                        id='slider-followers-bar',
                        className='border-top border-right border-left shadow-sm rounded-top px-2'
                    ),
                    dcc.Slider(
                        id='my-slider',
                        min=1,
                        max=50,
                        value=10,
                        marks={
                            3: '3',
                            5: '5',
                            10: '10',
                            20: '20',
                            50: '50'
                        },
                        className='border-bottom border-right border-left shadow-sm rounded-bottom p-4'
                    )
                ]),
                align='center',
                justify='center',
                className='mb-3'
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        figure=locationsMapPlot(data),
                        className='border shadow-sm rounded'
                    ),
                    align='center',
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(
                        id='slide-loc-bar',
                        className='border shadow-sm rounded px-2'
                    ),
                    align='center',
                    width=6
                )],
                align='center',
                justify='center',
                className='mb-3'
            ),
        ]),
        className='p-4 mb-3 bg-white rounded-bottom shadow'
    ),
])

# Tweets eda
tweets_layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H4('Informazioni sui tweets'),
            html.P(f"- Sono presenti {num_tweets} tweets unici.", className='px-3 mb-0'),
            html.P(f"- Il numero medio di parole per tweet è di {num_words} parole, senza preprocessing del testo.",
                    className='px-3'),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=sourcePie(data),
                        className='border shadow-sm rounded'
                    )
                ),
                align='center',
                justify='center'
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id='timeline',
                        className='border shadow-sm rounded'
                    )
                ),
                align='center',
                justify='center'
            ),
            dcc.Dropdown(
                id='dropdown-hashtag',
                options=[{'label': i, 'value': i} for i in hashtags]
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        figure=hashtagsBar(data),
                        className='border shadow-sm rounded'
                    ),
                    align='center',
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=topHashtagsBar(data),
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
                        figure=textWordcloud(data),
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
@app.callback(Output('tabs-eda-content', 'children'),
              [Input('tabs-eda', 'active_tab')])
def render_content(tab):
    if tab == 'tab-1-eda':
        return users_layout
    elif tab == 'tab-2-eda':
        return tweets_layout

# Bar plot slider
@app.callback(
    [Output('slider-followers-bar', 'figure'), Output('slider-unique-bar', 'figure'), Output('slide-loc-bar', 'figure')],
    [Input('my-slider', 'value')])
def updateFollowers(value):
    return [followersBar(data,value), uniqueUsersBar(data,value), locationsBar(data,value)]

# Date lineplot dropdown
@app.callback(
    Output('timeline', 'figure'),
    [Input('dropdown-hashtag', 'value')])
def updateTag(value):
    return tweetTimeLine(data,value)
