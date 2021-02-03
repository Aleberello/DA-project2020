import pandas as pd
import numpy as np
import random
import pickle

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

from app import app

## Data loading
data = pd.read_pickle("data/data.pkl")
data_tokens = pd.read_pickle("data/data_tokens.pkl")
vader_sent = pd.read_pickle("data/vader_sent.pkl")
roberta_sent = pd.read_pickle("data/roberta_sent.pkl")

# Sentiments extraction comparison
tabledata = vader_sent.join(roberta_sent, how='outer', lsuffix='_vader', 
                            rsuffix='_roberta').drop(columns=['text-clean_vader', 'text_roberta','text-clean_roberta'])
tabledata = tabledata.rename(columns={'text_vader':'text'})
table_pol = tabledata.copy()
tabledata['comparison'] = tabledata[['sentiment_vader', 'sentiment_roberta']].apply(
                                        lambda x: 'Equals' if x[0]==x[1] else 'Differents', axis=1)


table_pol['polarity_change'] = tabledata[['sentiment_vader', 'sentiment_roberta']].apply(
                                        lambda x: True if (x[0]=='Positive' and x[1]=='Negative') or 
                                        (x[0]=='Negative' and x[1]=='Positive') else False, axis=1)
table_pol = table_pol.drop(table_pol[table_pol['polarity_change']==False].index).drop(columns=['polarity_change'])

# Sentiment analysis table
data_sent = data.copy()
data_sent['sentiment'] = roberta_sent['sentiment']


## Functions
def sentBar(data1, data2):
    df1 = data1.sentiment.value_counts().reset_index()
    df1.columns=['SENTIMENT','COUNT']
    df1['PERC'] = np.around(((df1['COUNT'] / len(data1))*100),decimals=2)
    df2 = data2.sentiment.value_counts().reset_index()
    df2.columns=['SENTIMENT','COUNT']
    df2['PERC'] = np.around(((df2['COUNT'] / len(data2))*100),decimals=2)

    figBar = go.Figure()
    figBar.add_trace(go.Bar(
        x=df1['SENTIMENT'],
        y=df1['COUNT'],
        text=df1['PERC'],
        name='Vader',
        #marker_color='indianred'
    ))
    figBar.add_trace(go.Bar(
        x=df2['SENTIMENT'],
        y=df2['COUNT'],
        text=df2['PERC'],
        name='RoBERTa',
        #marker_color='lightsalmon'
    ))
    figBar.update_layout(barmode='group')
    figBar.update_traces(texttemplate='%{value} (%{text}%)', textposition='inside')

    return figBar

def tweetTimeLineSent(data_sent):
    df_sent = pd.concat([data_sent, pd.get_dummies(data_sent.sentiment)], axis=1)
    df_sent = df_sent.groupby(df_sent.date.dt.date).sum()

    figLine = go.Figure()
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Negative,
                        name='Negative',
                        mode="markers+lines",
                        line=dict(color='red')))
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Neutral,
                        name='Neutral',
                        line=dict(color='yellow')))
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Positive,
                        name='Positive',
                        line=dict(color='green')))
    return figLine


def textWordcloud(data, sent, num_words=100):

    def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl(0, 100%, {random.randint(25, 75)}%)" 
    def green_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl({random.randint(90, 150)}, 100%, 30%)" 
    def yellow_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl(42, 100%, {random.randint(25, 50)}%)" 
    
    df = data[(data_sent.sentiment==sent)]['tokens'].explode().value_counts()
    wc = WordCloud(width=400, height=800, background_color='white', max_words=num_words)
    wc.generate_from_frequencies(df)
    if sent=='Negative':
        fig = px.imshow(wc.recolor(color_func=red_color_func, random_state=3))
    elif sent=='Positive':
        fig = px.imshow(wc.recolor(color_func=green_color_func, random_state=3))
    else:
        fig = px.imshow(wc.recolor(color_func=yellow_color_func, random_state=3))
    fig.update_layout(title=f"Top-{num_words} parole nei {sent} tweets")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(hovermode=False)
    return fig


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
            dcc.Markdown('''
                Gli score estratti non sono tra loro direttamente comparabili: lo score di Vader varia tra \[-1,1\] 
                ed il sentiment viene scelto tramite soglie, lo score di RoBERTa invece varia tra 0 ed 1 ed è riferito 
                unicamente all\'etichetta estratta.
            '''),
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
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{comparison} = "Equals"',
                                    'column_id': 'comparison',
                                },
                                'backgroundColor': 'rgb(92, 184, 92)',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{comparison} = "Differents"',
                                    'column_id': 'comparison',
                                },
                                'backgroundColor': 'rgb(217, 83, 79)',
                                'color': 'white'
                            }
                        ],
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
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id='polarity-changes-table',
                        columns=[{"name": i, "id": i} for i in table_pol.columns],
                        data=table_pol.to_dict('records'),
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
                dbc.Col([
                    dbc.Row(
                        dbc.Col(
                            dcc.Slider(
                                id='wordcloud-slider',
                                min=10,
                                max=1000,
                                value=50,
                                step=10,
                                marks={
                                    10: '10',
                                    50: '50',
                                    100: '100',
                                    1000: '1000'
                                },
                                className='border shadow-sm rounded-bottom p-4'
                            )
                        )
                    ),
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(
                                id='wordcloud-neg',
                                className='border shadow-sm rounded'
                            ),
                            align='center',
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id='wordcloud-neu',
                                className='border shadow-sm rounded'
                            ),
                            align='center',
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id='wordcloud-pos',
                                className='border shadow-sm rounded'
                            ),
                            align='center',
                        )
                    ])
                ]),
                align='center',
                justify='center',
                className='mb-3'
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

# WordCloud words number slider
@app.callback(
    [Output('wordcloud-neg', 'figure'), Output('wordcloud-neu', 'figure'), Output('wordcloud-pos', 'figure')],
    [Input('wordcloud-slider', 'value')])
def updateNeg(value):
    return [textWordcloud(data_tokens, 'Negative', value),
            textWordcloud(data_tokens, 'Neutral', value),
            textWordcloud(data_tokens, 'Positive', value)]