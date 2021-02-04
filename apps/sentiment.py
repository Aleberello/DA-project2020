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
num_diff = tabledata['comparison'].value_counts()['Equals']

table_pol['polarity_change'] = tabledata[['sentiment_vader', 'sentiment_roberta']].apply(
                                        lambda x: True if (x[0]=='Positive' and x[1]=='Negative') or 
                                        (x[0]=='Negative' and x[1]=='Positive') else False, axis=1)
num_pol_diff = table_pol['polarity_change'].value_counts()[True]
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
    figBar.update_layout(barmode='group', title='Distribuzione sentiment dei due metodi di Sentiment Extraction')
    figBar.update_traces(texttemplate='%{value} (%{text}%)', textposition='inside')
    return figBar

def sentPie(data):
    df = data['sentiment'].value_counts().reset_index()
    df.columns=['SENTIMENT','COUNT']

    figPie = px.pie(df, values='COUNT', names='SENTIMENT', title='Distribuzione del sentiment nei tweets',
                    color='SENTIMENT',
                    color_discrete_map={'Positive':'#00CC96', 'Neutral':'#FECB52', 'Negative':'#EF553B'})
    return figPie

def tweetTimeLineSent(data_sent):
    df_sent = pd.concat([data_sent, pd.get_dummies(data_sent.sentiment)], axis=1)
    df_sent = df_sent.groupby(df_sent.date.dt.date).sum()

    figLine = go.Figure()
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Negative,
                        name='Negative',
                        mode="markers+lines",
                        line=dict(color='#EF553B')))
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Neutral,
                        name='Neutral',
                        mode="markers+lines",
                        line=dict(color='#FECB52')))
    figLine.add_trace(go.Scatter(x=df_sent.index, y=df_sent.Positive,
                        name='Positive',
                        mode="markers+lines",
                        line=dict(color='#00CC96')))

    figLine.update_layout(title='Andamento del sentiment nel tempo per numero di tweets')
    return figLine


def sentMapPlot(data):
    wmap = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
    df = data_sent[['country','sentiment']].value_counts().reset_index()
    df.columns=['CODE','SENTIMENT','TWEETS']
    temp = pd.DataFrame()
    for cod in df['CODE'].unique().tolist():
        tmp = df[df['CODE']==cod]
        temp = temp.append(tmp[tmp.TWEETS==tmp.TWEETS.max()])

    wmap_count = pd.merge(wmap, temp, how='left', on='CODE')
    wmap_count = wmap_count.dropna(subset=['SENTIMENT'])
    wmap_count['TWEETS'] = wmap_count['TWEETS'].fillna(0)

    figMap = px.choropleth(wmap_count, locations="CODE", color="SENTIMENT", hover_name="TWEETS",
                            color_discrete_map={'Positive':'#00CC96', 'Neutral':'#FECB52', 'Negative':'#EF553B'},
                            title='Distribuzione sentiment nelle nazioni per numero di tweets etichettati')
    figMap.update_coloraxes(showscale=False)
    return figMap

def sentFollowersBar(data):
    df = data[['user_name', 'user_followers']]
    df.columns=['USER','FOLLOWERS']
    df = df.groupby('USER').max().sort_values('FOLLOWERS', ascending=False).head(10).reset_index()
    tmp = []
    for user in df['USER'].unique().tolist():
        tmp.append(data[data['user_name']==user].sentiment.value_counts().sort_values(ascending=False).head(1).index[0])
    df['SENTIMENT']=tmp

    figBar = px.bar(df,
                    x='FOLLOWERS', y='USER', 
                    orientation='h', color='SENTIMENT',
                    text='FOLLOWERS', title="Top-10 account seguiti per numero di followers",
                    color_discrete_map={'Positive':'#00CC96', 'Neutral':'#FECB52', 'Negative':'#EF553B'})
    figBar.update_traces(texttemplate='%{value}', textposition='inside')
    figBar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=True)
    return figBar

def sentUsersBar(data):
    df = data[['user_name']].value_counts().head(10).reset_index()
    df.columns=['USER','TWEETS']
    tmp = []
    for user in df['USER'].unique().tolist():
        tmp.append(data[data['user_name']==user].sentiment.value_counts().sort_values(ascending=False).head(1).index[0])
    df['SENTIMENT']=tmp

    figBar = px.bar(df,
                    x='TWEETS', y="USER", 
                    orientation='h', color='SENTIMENT',
                    text='TWEETS', title="Top-10 account per numero di tweets",
                    color_discrete_map={'Positive':'#00CC96', 'Neutral':'#FECB52', 'Negative':'#EF553B'})
    figBar.update_traces(texttemplate='%{value}', textposition='inside')
    figBar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=True)
    return figBar

def textWordcloud(data, sent, title, num_words=100):

    def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl(0, 100%, {random.randint(25, 75)}%)" 
    def green_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl({random.randint(90, 150)}, 100%, 30%)" 
    def yellow_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return f"hsl(42, 100%, {random.randint(25, 50)}%)" 
    
    if title=='hashtags':
        df = data[(data_sent.sentiment==sent)]['hashtags'].explode().value_counts()
    else:
        df = data[(data_sent.sentiment==sent)]['tokens'].explode().value_counts()

    wc = WordCloud(width=400, height=800, background_color='white', max_words=num_words)
    wc.generate_from_frequencies(df)
    if sent=='Negative':
        fig = px.imshow(wc.recolor(color_func=red_color_func, random_state=3))
    elif sent=='Positive':
        fig = px.imshow(wc.recolor(color_func=green_color_func, random_state=3))
    else:
        fig = px.imshow(wc.recolor(color_func=yellow_color_func, random_state=3))
    fig.update_layout(title=f"Top-{num_words} {title} nei {sent} tweets")
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
            dcc.Markdown(f'''
                - Gli score estratti non sono tra loro direttamente comparabili: lo score di Vader varia tra \[-1,1\] 
                ed il sentiment viene scelto tramite soglie, lo score di RoBERTa invece varia tra 0 ed 1 ed è riferito 
                unicamente all\'etichetta estratta.  
                - Il numero di tweets aventi un sentiment diverso tra i due metodi è di {num_diff}, tra questi quelli 
                aventi un sentiment diametralmente opposto è di {num_pol_diff}.
            '''),
            dbc.Row(
                dbc.Col([
                    html.H5("Comparativa sentiment estratto dai due metodi", className='mb-2'),
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
                    )],
                    className="p-1 mb-3 border shadow-sm bg-white rounded"
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=sentBar(vader_sent,roberta_sent),
                        className='border shadow-sm rounded px-2 mb-3'
                    ),
                ),
            ),
            dbc.Row(
                dbc.Col([
                    html.H5("Comparativa sentiment estratto dai due metodi opposto", className='mb-2'),
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
                    )],
                    className="p-1 mb-3 border shadow bg-white rounded"
                )
            )
        ]),
        className='p-4 mb-3 bg-white rounded-bottom shadow'
    ),
])




# Sentiment analysis
sent_analy_layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H4('Analisi sul sentiment estratto'),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=sentPie(data_sent),
                        className='border shadow-sm rounded px-2'
                    )
                ),
                align='center',
                justify='center',
                className='mb-3'
            ),         
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id='timeline-sent',
                        figure=tweetTimeLineSent(data_sent),
                        className='border shadow-sm rounded px-2'
                    )
                ),
                align='center',
                justify='center',
                className='mb-3'
            ),
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id='table-text-linechart-sent',
                        columns=[{"name": 'date', "id": 'date'}, {"name": 'text', "id": 'text'},
                                {'name':'sentiment', 'id':'sentiment'}],
                        style_table={
                            'overflowX' : 'auto',
                            'width': '100%',
                        },
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{sentiment} = "Positive"',
                                    'column_id': 'sentiment',
                                },
                                'backgroundColor': '#00CC96',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{sentiment} = "Neutral"',
                                    'column_id': 'sentiment',
                                },
                                'backgroundColor': '#FECB52',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{sentiment} = "Negative"',
                                    'column_id': 'sentiment',
                                },
                                'backgroundColor': '#EF553B',
                                'color': 'white'
                            }
                        ],
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
                    className="p-2 border shadow-sm rounded"
                ),
                align='center',
                justify='center',
                className='mb-3'
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=sentMapPlot(data_sent),
                        className='border shadow-sm rounded px-2'
                    ),
                ),
                className='mb-3'
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=sentFollowersBar(data_sent),
                        className='border shadow-sm rounded px-2'
                    ),
                ),
                className='mb-3'
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        figure=sentUsersBar(data_sent),
                        className='border shadow-sm rounded px-2'
                    ),
                ),
                className='mb-3'
            ),
            dbc.Row(
                dbc.Col([
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
                    ]),
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
                                id='wordcloud-neg-hash',
                                className='border shadow-sm rounded'
                            ),
                            align='center',
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id='wordcloud-neu-hash',
                                className='border shadow-sm rounded'
                            ),
                            align='center',
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id='wordcloud-pos-hash',
                                className='border shadow-sm rounded'
                            ),
                            align='center',
                        )
                    ]),
                ]),
                align='center',
                justify='center',
                className='mb-3'
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
    [Output('wordcloud-neg', 'figure'), Output('wordcloud-neu', 'figure'), Output('wordcloud-pos', 'figure'),
    Output('wordcloud-neg-hash', 'figure'), Output('wordcloud-neu-hash', 'figure'), Output('wordcloud-pos-hash', 'figure')],
    [Input('wordcloud-slider', 'value')])
def updateNeg(value):
    return [textWordcloud(data_tokens, 'Negative', 'parole', value),
            textWordcloud(data_tokens, 'Neutral', 'parole', value),
            textWordcloud(data_tokens, 'Positive', 'parole', value),
            textWordcloud(data, 'Negative','hashtags', value),
            textWordcloud(data, 'Neutral','hashtags', value),
            textWordcloud(data, 'Positive', 'hashtags', value)]

# Date lineplot text field table
@app.callback(
    Output('table-text-linechart-sent', 'data'),
    Input('timeline-sent', 'selectedData'))
def displaySelectedData(selectedData):
    if selectedData is None:
        new_data = data_sent[['date','text','sentiment']].to_dict('records')
    else:
        date1 = selectedData['range']['x'][0]   
        date2 = selectedData['range']['x'][1]
        new_data = data_sent.set_index('date').loc[date1:date2][['text','sentiment']].sort_index().reset_index()[['date','text','sentiment']]
        new_data['date'] = new_data['date'].dt.date
        new_data = new_data.to_dict('records')
    return new_data