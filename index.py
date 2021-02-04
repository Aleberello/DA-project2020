import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import home, eda, sentiment


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
            children=[
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("EDA", href="/eda", active="exact"),
                dbc.NavLink("Sentiment Analysis", href="/sentiment-analysis", active="exact")
            ],
            brand="COVID19 Vaccine Tweets",
            color="dark",
            dark=True,
            sticky='Top',
            className='shadow mb-3'
        ),
    dbc.Spinner(id="loading-content", fullscreen=True),
    html.Footer(
        dbc.Row([
            dbc.Col(
                html.Img(
                    src='https://upload.wikimedia.org/wikipedia/it/thumb/7/7d/Logo_Universit%C3%A0_Milano-Bicocca.svg/1200px-Logo_Universit%C3%A0_Milano-Bicocca.svg.png',
                    style={
                        'max-width' : '30%',
                        'padding-top' : 0,
                        'padding-right' : 0
                    }
                ),
                width=1,
                align='center',
                className='m-0 p-0 text-right'
            ),
            dbc.Col([
                html.P('Università degli Studi di Milano-Bicocca', className='text-light mb-0 small'),
                html.P('Alessandro Bertolo (808314) - a.bertolo2@campus.unimib.it', className='text-light mb-0 small')
            ],
            align='center'
            )
        ]),
        className="bg-dark px-5 py-3 position-sticky bottom-0",
    ),
], className="bg-light")


## Callbacks
@app.callback(Output('loading-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/eda':
        return eda.layout
    elif pathname == '/sentiment-analysis':
        return sentiment.layout
    else:
        return dbc.Jumbotron([
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised...")
        ])
        

if __name__ == '__main__':
    app.run_server(debug=True)