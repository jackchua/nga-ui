# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018

@author: jimmybow
"""
import plotly.express as px
from datetime import datetime
from dash import Dash
from dash.dependencies import Input, State, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd

from .utility import apply_layout_with_auth, load_object, save_object, get_postgres_sqlalchemy_uri

# global vars
_URL_BASE = '/dash/google_ads_budget_optimizer/'
_ACCOUNT_VALUES = ('test')

# layout
layout = html.Div([
    html.H2('Select account to optimize budget for'),
    dcc.Dropdown(
        id='account-dropdown',
        options=[
            {'label': 'Test', 'value':'test'},
            {'label': 'Hult', 'value':'hult'}
        ],
        value='test'
    ),
    html.H2('Review and submit proposed optimization changes'),
    dash_table.DataTable(
        id='campaign-budget-table',
        columns=(
            [{'id': 'c_id', 'name': 'Campaign ID'},
             {'id': 'old_value', 'name': 'Last Known Daily Budget ($)'},
             {'id': 'new_value', 'name': 'Proposed Daily Budget ($)'},
             {'id': 'readable_time', 'name': 'Create Time'}]
        ),
        editable=True
    ),
    html.Button('Submit', id='submit-optimization-change', type='submit'),
    html.H2('Simulated impact of optimization change'),
    html.H4('TBD'),
    html.H2('Budget changes over time per campaign'),
    dcc.Graph(id='budget-over-time-graph'),
    html.Div(id='intermediate-value', style={'display': 'none'})
], style={'width': '500'})

# main initialization and callbacks
def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=_URL_BASE)
    pgdb = create_engine(get_postgres_sqlalchemy_uri())
    apply_layout_with_auth(app, layout)

    @app.callback(Output('intermediate-value', 'children'),
                         [Input('account-dropdown', 'value')])
    def get_campaign_budget_data(value):
        statement = text("""
        select * from {}.google_ads_budget_bid_staging where bid_unit='campaign' and bid_type='budget';
        """.format(value))
        with pgdb.connect() as con:
            rs = con.execute(statement)
        data = rs.fetchall()
        keys = rs.keys()
        df = pd.DataFrame(data, columns=keys)
        df.c_id = df.c_id.astype(str)
        df.old_value = df.old_value / 1000000000.
        df.new_value = df.new_value / 1000000000.
        df['readable_time'] = df['created_at'].dt.strftime('%Y-%m-%d | %H:%M:%S').replace('T',' ')
        return df.to_json(date_format='iso', orient='split')


    @app.callback(Output('campaign-budget-table', 'data'),
                  [Input('intermediate-value', 'children')])
    def update_campaign_budget_table(json_data):
        df = pd.read_json(json_data, orient='split')

        # get the latest unsubmitted rows for each campaign
        df = df[df.submitted==False]
        idx = df.groupby(['c_id'])['created_at'].transform(max) == df['created_at']
        df = df[idx]
        columns = ['c_id', 'old_value', 'new_value', 'readable_time']
        return df[columns].to_dict(orient='records')

    @app.callback(Output('budget-over-time-graph', 'figure'),
                  [Input('intermediate-value', 'children')])
    def update_budget_over_time_graph(json_data):
        df = pd.read_json(json_data, orient='split')
        df = df[df.submitted==True]
        df['c_id'] = df['c_id'].astype(str)

        # return the graph
        return px.scatter(
            df,
            x='created_at',
            y='new_value',
            color='c_id',
            width=1000
        )\
            .update_traces(mode='lines+markers')\
            .update_layout(legend_title='<b> Campaign ID </b>')

    return app.server