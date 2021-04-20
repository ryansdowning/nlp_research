"""This module implements the dash plotly app for the alpha-sentiment dashboard, all data is preloaded into memory for
quick display of various graph settings selected by the user
"""
import datetime
from functools import partial

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from loguru import logger

from finlang.nlp_utils import db_utils as dbu
from finlang.plotting import plotting_utils as pu

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
MODELS = ['random_model', 'test_model', 'distilbert-base-uncased-finetuned-sst-2-english']
DATA, LINKING = dbu.get_sentiment_by_keywords(MODELS, cols=('id', 'source', 'created_utc'), return_linking=True)
INITIAL_KEYWORDS = ['FB', 'AAPL', 'NFLX', 'GOOG', 'AMZN']
INITIAL_SOURCES = ['r/investing', 'r/stocks', 'r/wallstreetbets']
SOURCES = set(DATA['source'])
INITIAL_MODEL = 'random_model'
CLICK_DELAY = 5
update_data = partial(
    pu.compute_facet_data, data=DATA, keywords_linking=LINKING,
    date_col='created_utc', source_col='source'
)
update_plot = partial(pu.plot_keywords, date_col='created_utc')

app.layout = html.Div([
    html.H1("Alpha Sentiment"),
    html.Div([
        html.Label('Date Range'),
        dcc.DatePickerRange(
            id='date-range-select',
            min_date_allowed=DATA['created_utc'].min(),
            max_date_allowed=DATA['created_utc'].max(),
            initial_visible_month=datetime.datetime(2019, 1, 1),
            start_date=datetime.datetime(2019, 1, 1),
            end_date=datetime.datetime(2020, 1, 1)
        ),
        html.Label('Data Frequency'),
        dcc.Dropdown(
            id='frequency-dropdown',
            options=[
                {'label': 'Daily', 'value': 'D'},
                {'label': 'Weekly', 'value': 'W'},
                {'label': 'Monthly', 'value': 'M'},
                {'label': 'Quarterly', 'value': 'Q'},
                {'label': 'Yearly', 'value': 'Y'}
            ],
            value='M',
        ),
        html.Label('Keywords'),
        dcc.Dropdown(
            id='keywords-dropdown',
            options=[{'label': keyword, 'value': keyword} for keyword in LINKING.keys()],
            value=[keyword for keyword in INITIAL_KEYWORDS if keyword in LINKING],
            multi=True,
            placeholder="Select keywords to include in plots"
        ),
        html.Label('Sources'),
        dcc.Dropdown(
            id='sources-dropdown',
            options=[{'label': source, 'value': source} for source in SOURCES],
            value=[source for source in INITIAL_SOURCES if source in SOURCES],
            multi=True,
            placeholder="Select sources to include in plots"
        ),
        html.Label('Sentiment Model'),
        dcc.Dropdown(
            id='models-dropdown',
            options=[{'label': model, 'value': model} for model in MODELS],
            value=INITIAL_MODEL,
            placeholder="Select sentiment model to use for determining sentiment values"
        ),
        html.Label("Summary Type"),
        dcc.RadioItems(
            id='summary-type',
            options=[
                {'label': 'Mentions', 'value': 'mentions'},
                {'label': 'Count', 'value': 'count'},
                {'label': 'Sum', 'value': 'sum'}
            ],
            value='mentions',
        ),
        html.Label('Additional Options'),
        dcc.Checklist(
            id='add-options',
            options=[
                {'label': 'Separate Keywords', 'value': 'SK'},
                {'label': 'Separate Sources', 'value': 'SS'},
                {'label': 'Overlay Lines', 'value': 'OL'},
            ],
            value=['SK', 'SS'],
            labelStyle={'display': 'inline-block'}
        ),
        html.Button('Update', id='update-button', n_clicks=0),
    ]),
    dcc.Tabs([
        dcc.Tab(label='Graphs', children=[
            html.Div([dcc.Loading(
                id="loading",
                children=[dcc.Graph(id='main-fig', style={'height': '80vh'})],
                type='default'
            )])
        ]),
        dcc.Tab(label='Data', children=[
            dash_table.DataTable(
                id='summary-table',
                page_size=100,
                style_table={'height': '80vh', 'overflowY': 'auto'},
                sort_action='native',
                filter_action='native'
            )
        ])
    ]),
    html.Div([
        html.A(
            id='github-link', href='https://github.com/ryansdowning/nlp_research', target='_blank',
            children=html.Img(
                id='github-image', src='https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg',
                width=40, height=40
            )
        ),
        html.A(
            id='bmac-link', href='https://www.buymeacoffee.com/ryansd', target='_blank',
            children=html.Img(
                id='bmac-image', src="https://i.ibb.co/x6KTCf9/bmc-icon-black.png",
                width=58, height=40
            )
        )
    ]),
    dcc.Store(
        id='session', storage_type='session', data={'last_click': datetime.datetime.now().timestamp() - CLICK_DELAY}
    )
])


@app.callback(Output('session', 'data'), Input('update-button', 'n_clicks'), State('session', 'data'))
def update_button_timer(n_clicks, data):
    """Checks if enough time has passed between update requests

    Args:
        n_clicks: how many updates requested by user
        data: dictionary of json data stored in user's browser - in this case it stores the time the last click was
              requested, and it was accepted.

    Raises
        dash.exceptions.PreventUpdate: If not enough time has passed between requests and updates

    Returns:
        Dictionary with key 'last_click' and the timestamp of the last click that was allowed
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    curr = datetime.datetime.now().timestamp()
    if data['last_click'] <= curr - CLICK_DELAY:
        return {'last_click': curr}
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output('main-fig', 'figure'),
    Output('summary-table', 'data'),
    Output('summary-table', 'columns'),
    Input('session', 'data'),
    State('date-range-select', 'start_date'),
    State('date-range-select', 'end_date'),
    State('frequency-dropdown', 'value'),
    State('keywords-dropdown', 'value'),
    State('sources-dropdown', 'value'),
    State('models-dropdown', 'value'),
    State('summary-type', 'value'),
    State('add-options', 'value')
)
def update_figure(data, start, end, freq, keywords, sources, model, summary, options):
    """Updates figure and data given the current user selected settings"""
    logger.info(str((data['last_click'], start, end, freq, keywords, sources, model, summary, options)))
    facet_keywords = 'SK' in options
    facet_sources = 'SS' in options
    overlap = 'OL' in options
    facet_data = update_data(
        start_date=start, end_date=end, label_col=model, sources=sources, keywords=keywords, frequency=freq,
        summary=summary, facet_sources=facet_sources, facet_keywords=facet_keywords
    )
    data_columns = [{'name': col, 'id': col} for col in facet_data.columns]
    fig = update_plot(
        facet_data=facet_data, summary=summary, facet_keywords=facet_keywords,
        facet_sources=facet_sources, overlap=overlap, label_col=model
    )
    return fig, facet_data.to_dict(orient='records'), data_columns


if __name__ == '__main__':
    app.run_server()
