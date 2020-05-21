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
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from .utility import apply_layout_with_auth, load_object, save_object, get_postgres_sqlalchemy_uri

# global vars
_URL_BASE = '/dash/google_ads_ltv_optimizer/'
_ACCOUNT_VALUES = ('test')

# layout
layout = html.Div([
    html.H2('Select account to assess LTV model'),
    dcc.Dropdown(
        id='account-dropdown',
        options=[
            {'label': 'Hult', 'value':'hult'}
        ],
        value='hult'
    ),
    html.H2('LTV model metrics'),
    html.Div([
        html.H3('Pick model type:'),
        dcc.Dropdown(id='ltv-metrics-model-type-dropdown')
    ]),
    dcc.Graph(id='ltv-metrics-graph'),
    html.H2('LTV sensitivities'),
    dcc.Dropdown(id='sensitivities-date-dropdown'),
    html.Br(),
    dash_table.DataTable(
        id='ltv-sensitivities-table',
        columns=(
            [{'id': 'model_type', 'name': 'Model Type'},
             {'id': 'model_date', 'name': 'Create Date'},
             {'id': 'category', 'name': 'Feature Category'},
             {'id': 'category_count', 'name': 'Feature Count'},
             {'id': 'category_value', 'name': 'Feature Value'},
             {'id': 'sensitivity', 'name': 'Feature Sensitivity'}]
        ),
        editable=False
    ),
], style={'width': '500'})

# main initialization and callbacks
def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=_URL_BASE)
    pgdb = create_engine(get_postgres_sqlalchemy_uri())
    apply_layout_with_auth(app, layout)

    @app.callback([Output('sensitivities-date-dropdown', 'options'), Output('sensitivities-date-dropdown', 'value')],
                  [Input('account-dropdown', 'value')])
    def update_date_dropdowns(account_value):
        statement = text("""
        select distinct model_date from {}.ltv_model_sensitivities order by model_date;
        """.format(account_value))
        with pgdb.connect() as con:
            rs = con.execute(statement)
            data = rs.fetchall()
            data = [
                {'label': datetime.strftime(d[0], "%Y-%m-%d"), 'value': datetime.strftime(d[0], "%Y-%m-%d")}
                for d in data
            ]
            default = data[0]['value']
        return [data, default]

    @app.callback([Output('ltv-metrics-model-type-dropdown', 'options'), Output('ltv-metrics-model-type-dropdown', 'value')],
                  [Input('account-dropdown', 'value')])
    def update_ltv_metrics_model_type_dropdown(account_value):
        statement = text("""
        select distinct model_type from {}.ltv_model_metrics;
        """.format(account_value))
        with pgdb.connect() as con:
            rs = con.execute(statement)
            data = rs.fetchall()
            data = [
                {'label': d[0], 'value': d[0]}
                for d in data
            ]
            print(data)
            default = data[0]['value']
        return [data, default]

    @app.callback(
        Output('ltv-metrics-graph', 'figure'),
        [Input('account-dropdown', 'value'), Input('ltv-metrics-model-type-dropdown', 'value')]
    )
    def get_ltv_metrics_graph(account_value, model_type):
        statement = text("""
        select * from {}.ltv_model_metrics where model_type='{}' order by model_type, model_date
        """.format(account_value, model_type))
        with pgdb.connect() as con:
            rs = con.execute(statement)
            data = rs.fetchall()
            keys = rs.keys()
            df = pd.DataFrame(data, columns=keys)

        df = pd.melt(
            df, id_vars=['model_date','model_type'],
            value_vars=['precision','recall','accuracy', 'auc']
        )
        df = df.rename(columns={'variable':'metric'})
        fig = px.scatter(
            df,
            x='model_date',
            y='value',
            color='metric'
        )
        return fig

    @app.callback(
        Output('ltv-sensitivities-table', 'data'),
        [Input('account-dropdown','value'), Input('sensitivities-date-dropdown', 'value')]
    )
    def get_ltv_sensitivities_table(account_value, sensitivity_date):
        statement = text("""
        select
             category,
             category_value,
             category_count,
             ROUND(sensitivity::numeric, 3) sensitivity,
             model_type,
             model_date
        from {}.ltv_model_sensitivities where model_date = '{}'
        """.format(account_value, sensitivity_date)) # assume already ordered by batch process
        with pgdb.connect() as con:
            rs = con.execute(statement)
        data = rs.fetchall()
        keys = rs.keys()
        df = pd.DataFrame(data, columns=keys)
        print(df)
        return df.to_dict(orient='records')


    return app.server