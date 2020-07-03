# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018
"""
import plotly.express as px
import json
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, State, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from time import sleep
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from econml.dml import LinearDMLCateEstimator, NonParamDMLCateEstimator, ForestDMLCateEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.inference import BootstrapInference
import os
import warnings

from .utility import apply_layout_with_auth, load_object, save_object, get_postgres_sqlalchemy_uri

# global vars
_URL_BASE = '/dash/google_ads_causal_inference/'
_ACCOUNT_VALUES = ('hult')

# layout
layout = html.Div([
    html.H4('Google Ads account to run analysis for'),
    dcc.Dropdown(
        id='account-dropdown',
        options=[
            {'label': 'Hult PG MA', 'value': '410-031-1524'},
            {'label': 'Hult PG MBA', 'value': '946-498-3032'},
        ],
        value='946-498-3032'
    ),
    html.H4('Type in campaign IDs in your account in the control group, along with start and end dates'),
    dash_table.DataTable(
        id='control-campaigns-table',
        columns=(
            [{'id': 'campaign_id', 'name': 'Campaign ID'},
             {'id': 'start_date', 'name': 'Start Date'},
             {'id': 'end_date', 'name': 'End Date'}]
        ),
        data=[
            {'campaign_id': 'xxxyyzzz', 'start_date':'2020-06-01', 'end_date': '2020-06-10'}
        ],
        editable=True,
        row_deletable=True
    ),
    html.Button('Add Row', id='control-campaigns-row-button', n_clicks=0),
    html.H4('Type in campaign IDs in your account in the test group, along with start and end dates'),
    dash_table.DataTable(
        id='test-campaigns-table',
        columns=(
            [{'id': 'campaign_id', 'name': 'Campaign ID'},
             {'id': 'start_date', 'name': 'Start Date'},
             {'id': 'end_date', 'name': 'End Date'}]
        ),
        data=[
            {'campaign_id': 'xxxyyzzz', 'start_date':'2020-06-01', 'end_date': '2020-06-10'}
        ],
        editable=True,
        row_deletable=True
    ),
    html.Button('Add Row', id='test-campaigns-row-button', n_clicks=0),
    html.H4('Select features you want to use to build your baseline:'),
    dcc.Dropdown(
        id='baseline-features',
        options=[
            {'label': 'Impressions + Lags', 'value': 'impressions'},
            {'label': 'Clicks + Lags', 'value': 'clicks'},
            {'label': 'Conversions', 'value': 'conversions'},
            {'label': 'Bounce Rate + Lags', 'value': 'bounce_rate'},
            {'label': 'Average Time on Site + Lags', 'value': 'average_time_on_site'},
            {'label': 'Day of Week', 'value': 'dow'},
            {'label': 'Day of Month', 'value': 'day'},
            {'label': 'Week Number', 'value': 'week'},
            {'label': 'Month', 'value': 'month'},
            {'label': 'Year', 'value': 'year'}
        ],
        value=['impressions', 'clicks', 'day', 'dow'],
        multi=True
    ),
    html.Br(),
    html.Button(
        'Run Analysis', id='run-analysis-button', type='submit', className='btn btn-alert',
        style={'align':'center'}, n_clicks=0
    ),
    html.Hr(),
    html.Div([
        html.Div([
            html.H4('Outcome histograms'),
            dcc.Dropdown(
                id='experiment-outcome-to-plot', multi=False),
            dcc.Graph(id='experiment-outcome-histogram'),
        ], style={'display':'inline-block', 'width':'49%'}),
        html.Div([
            html.H4('Experiment effects'),
            dcc.Graph(id='experiment-box-plot')
        ], style={'display':'inline-block', 'width':'49%'})
    ])
], style={'width': '500'})

# main initialization and callbacks
def Add_Dash(server):
    app = Dash(
        server=server, url_base_pathname=_URL_BASE
    )
    pgdb = create_engine(get_postgres_sqlalchemy_uri())
    apply_layout_with_auth(app, layout)

    @app.callback(
        Output('control-campaigns-table', 'data'),
        [Input('control-campaigns-row-button', 'n_clicks')],
        [State('control-campaigns-table', 'data'),
         State('control-campaigns-table', 'columns')])
    def add_row_control_campaigns(n_clicks, rows, columns):
        if n_clicks > 0:
            rows.append({c['id']: '' for c in columns})
        return rows

    @app.callback(
        Output('test-campaigns-table', 'data'),
        [Input('test-campaigns-row-button', 'n_clicks')],
        [State('test-campaigns-table', 'data'),
         State('test-campaigns-table', 'columns')])
    def add_row_test_campaigns(n_clicks, rows, columns):
        if n_clicks > 0:
            rows.append({c['id']: '' for c in columns})
        return rows

    # click for in progress
    @app.callback(
        [Output('run-analysis-button', 'disabled')],
        [Input('run-analysis-button', 'n_clicks')]
    )
    def gray_out_submit(n_clicks):
        if n_clicks > 0:
            return [True]

    # giant callback for experimental results
    @app.callback(
        [Output('experiment-outcome-to-plot', 'options'),
         Output('experiment-outcome-histogram', 'data'),
         Output('experiment-box-plot', 'data')],
        [Input('run-analysis-button', 'n_clicks')],
        [State('account-dropdown', 'value'),
         State('control-campaigns-table', 'value'),
         State('test-campaigns-table', 'value')]
    )
    def compute_experimental_results(n_clicks, account_string, control_data, test_data):
        if n_clicks > 0:
            print(account_string)
            sleep(10)
            return [[],[],[]]

        # query = """
        #     SELECT * FROM google_ads_hult.campaign_performance_reports where campaign_id in ('{}', '{}')
        # """.format(control_campaign, test_campaign)
        # data = dblayer.query_db(query, output=True)

    return app.server