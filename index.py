import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import home, eda, sentiment, misc


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
            children=[
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("EDA", href="/eda", active="exact"),
                dbc.NavLink("Sentiment analysis", href="/sentiment-analysis", active="exact"),
                dbc.NavLink("Misc", href="/misc", active="exact")
            ],
            brand="COVID19 Vaccine Tweets",
            color="dark",
            dark=True,
            sticky='Top',
            className='shadow mb-3'
        ),
    dbc.Spinner(id="loading-content", fullscreen=True)
], className="bg-light")

@app.callback(Output('loading-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/eda':
        return eda.layout
    elif pathname == '/sentiment-analysis':
        return sentiment.layout
    elif pathname == '/misc':
        return misc.layout
    else:
        return dbc.Jumbotron([
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised...")
        ])
        

if __name__ == '__main__':
    app.run_server(debug=True)