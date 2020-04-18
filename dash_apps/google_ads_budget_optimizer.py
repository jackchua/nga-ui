# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018

@author: jimmybow
"""
from dash import Dash
from dash.dependencies import Input, State, Output
from .utility import apply_layout_with_auth, load_object, save_object, get_postgres_sqlalchemy_uri
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd

_URL_BASE = '/dash/google_ads_budget_optimizer/'
_ACCOUNT_VALUES = ('test')

layout = html.Div([
    dcc.Input(id='my-id', value='Account to optimize (i.e. test)', type='text'),
    dcc.Graph(id='my-graph')
], style={'width': '500'})

def Add_Dash(server):
    app = Dash(server=server, url_base_pathname=_URL_BASE)
    pgdb = create_engine(get_postgres_sqlalchemy_uri())
    apply_layout_with_auth(app, layout)

    @app.callback(Output('my-graph', 'figure'), [Input(component_id='my-id', component_property='value')])
    def update_graph(input_value):
        if input_value in _ACCOUNT_VALUES:
            statement = text("""
            select * from {}.google_ads_budget_bid_staging where submitted=true
            """.format(input_value))
            with pgdb.connect() as con:
                rs = con.execute(statement)
            data = rs.fetchall()
            keys = rs.keys()
            df = pd.DataFrame(data, columns=keys)
            df.c_id = df.c_id.astype(str)

            # df = pd.DataFrame([(1,1),(2,2),(3,3),(4,4)], columns=['x','y'])

            print(df)

            return {
                'data': [{
                    'x': df.created_at,
                    'y': df.new_value,
                    'color': df.c_id,
                    'type': 'line'
                }],
                'layout': {'margin': {'l': 40, 'r':0, 't':20, 'b':30}}
            }

        else:
            # not a valid account
            pass


    return app.server