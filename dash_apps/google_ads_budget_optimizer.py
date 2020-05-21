# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018
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
import numpy as np

from .utility import apply_layout_with_auth, load_object, save_object, get_postgres_sqlalchemy_uri

# global vars
_URL_BASE = '/dash/google_ads_budget_optimizer/'
_ACCOUNT_VALUES = ('test')

def func(x_vec):
    # TODO: change hardcoded values with an actual database call to coefficients
    a_vec=np.array([387.25692279125417, 86.67476346952401])
    b_vec=np.array([66.66793218354748, 37.42248784994848])
    c_vec=np.array([0.0004012805883538062, 0.0036356668160448602])
    return np.sum(a_vec - np.multiply(a_vec-b_vec, np.exp(np.multiply(-c_vec,x_vec))))

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
    html.H2('(Optional) Populate table with a pre-analyzed budget for your campaigns'),
    dcc.Slider(
        id='budget-slider'
    ),
    html.Div(id='slider-output-text', children='Total budget:'),
    html.Div(id='slider-output-container', style={'padding-bottom':'100px'}),
    html.H2('Review and submit proposed optimization changes'),
    dash_table.DataTable(
        id='campaign-budget-table',
        columns=(
            [{'id': 'c_id', 'name': 'Campaign ID'},
             {'id': 'old_value', 'name': 'Last Known Daily Budget ($)'},
             {'id': 'new_value', 'name': 'Proposed Optimal Daily Budget ($)'},
             {'id': 'readable_time', 'name': 'Create Time'}]
        ),
        editable=True
    ),
    html.Button('Submit to Google', id='submit-optimization-change', type='submit'),
    html.H2('Simulated impact of optimization change'),
    dcc.Graph(id='opt-simulation-graph'),
    html.H2('Budget changes over time per campaign'),
    # dcc.Graph(id='budget-over-time-graph'),
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
        df.old_value = df.old_value
        df.new_value = df.new_value
        df['readable_time'] = df['created_at'].dt.strftime('%Y-%m-%d | %H:%M:%S').replace('T',' ')
        return df.to_json(date_format='iso', orient='split')

    @app.callback([Output('budget-slider', 'max'), Output('budget-slider', 'step'), Output('budget-slider', 'min'), Output('budget-slider', 'marks'), Output('budget-slider', 'value')],
                  [Input('intermediate-value', 'children')])
    def update_budget_slider(json_data):
        df = pd.read_json(json_data, orient='split')
        budgets = df.groupby('solution_id')['new_value'].sum().values
        budgets = [int(b) for b in budgets]
        max_budget = np.max(budgets)
        min_budget = np.min(budgets)
        marks = {i:{'label': '${}'.format(i), 'style': {'fontSize': 14,'writing-mode': 'vertical-rl','text-orientation': 'upright'}} for idx, i in enumerate(budgets)}
        return [max_budget, None, 0.0, marks, min_budget]

    @app.callback(Output('slider-output-container', 'children'),
                  [Input('budget-slider', 'value')])
    def update_budget_output(value):
        return '${}'.format(value)

    @app.callback(Output('campaign-budget-table', 'data'),
                  [Input('budget-slider', 'value'), Input('intermediate-value', 'children')])
    def update_campaign_budget_table(selected_budget, json_data):
        df = pd.read_json(json_data, orient='split')

        # get the solution with the closest budget value
        budgets = df.groupby('solution_id')['new_value'].sum().reset_index()
        budgets['abs'] = np.abs(float(selected_budget)-budgets['new_value'])
        best_sol_id = budgets.iloc[budgets['abs'].argmin()]['solution_id']
        print(best_sol_id)

        # get the latest unsubmitted rows for each campaign
        df = df[(df.submitted==False) & (df.solution_id==best_sol_id)]
        idx = df.groupby(['c_id'])['created_at'].transform(max) == df['created_at']
        df = df[idx]
        columns = ['c_id', 'old_value', 'new_value', 'readable_time']
        return df[columns].to_dict(orient='records')

    @app.callback(Output('opt-simulation-graph', 'figure'),
                  [Input('intermediate-value', 'children'), Input('campaign-budget-table', 'data')])
    def update_opt_simulation_graph(json_data, table_data):
        sim = pd.read_json(json_data, orient='split')
        df = pd.DataFrame(table_data)

        # first plot optimal simulated curve
        sim = sim[['solution_id', 'solution_clicks', 'solution_budget']].drop_duplicates()
        sim.columns = ['solution_id', 'total_clicks', 'total_budget']
        sim['proposal'] = 'efficient frontier'
        fig = px.scatter(
            sim,
            x='total_budget',
            y='total_clicks',
            color='proposal',
            width=800
        )

        # now take any potential manual input and plot as red
        man_cost_vec = df['new_value'].astype('float').values
        man_cost = sum(man_cost_vec)
        man_clicks = func(man_cost_vec)
        man = pd.DataFrame([('manual', man_cost, man_clicks)], columns=['proposal', 'total_budget', 'total_clicks'])
        fig2 = px.scatter(
            man,
            x='total_budget',
            y='total_clicks',
            color='proposal',
            width=800
        ).update_traces(
            mode='markers',
            marker_symbol='diamond',
            marker_color='red',
            marker_size=15
        )
        fig.add_trace(fig2.data[0])
        return fig
    #
    # @app.callback(Output('budget-over-time-graph', 'figure'),
    #               [Input('intermediate-value', 'children')])
    # def update_budget_over_time_graph(json_data):
    #     df = pd.read_json(json_data, orient='split')
    #     df = df[df.submitted==True]
    #     df['c_id'] = df['c_id'].astype(str)
    #
    #     # return the graph
    #     return px.scatter(
    #         df,
    #         x='created_at',
    #         y='new_value',
    #         color='c_id',
    #         width=1000
    #     )\
    #         .update_traces(mode='lines+markers')\
    #         .update_layout(legend_title='<b> Campaign ID </b>')

    return app.server