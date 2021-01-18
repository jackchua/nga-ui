# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from dash_apps.utility import apply_layout_with_auth, load_object, save_object, get_postgres_sqlalchemy_uri
from pandas import DataFrame
from datetime import datetime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

SCORING_FILE = "data/gat_ltv_precitions_20210111.csv"
#
# metrics = ['all_acc', 'all_balanced_acc', 'all_roc_auc', 'all_precision', 'all_recall', 'all_pr_auc',
#            'all_lift', 'all_mae', 'all_rmse', 'since_three_month_ago_acc',
#            'since_three_month_ago_balanced_acc', 'since_three_month_ago_roc_auc',
#            'since_three_month_ago_precision', 'since_three_month_ago_recall',
#            'since_three_month_ago_pr_auc', 'since_three_month_ago_lift', 'since_three_month_ago_mae',
#            'since_three_month_ago_rmse', 'scoring_since_six_month_ago_acc',
#            'scoring_since_six_month_ago_balanced_acc', 'scoring_since_six_month_ago_roc_auc',
#            'scoring_since_six_month_ago_precision', 'scoring_since_six_month_ago_recall',
#            'scoring_since_six_month_ago_pr_auc', 'scoring_since_six_month_ago_lift',
#            'scoring_since_six_month_ago_mae', 'scoring_since_six_month_ago_rmse',
#            'mean_error_old_clients', 'true_positive_old_clients', 'mean_error_new_clients',
#            'true_positive_new_clients', 'since_one_month_ago_acc', 'since_one_month_ago_balanced_acc',
#            'since_one_month_ago_roc_auc', 'since_one_month_ago_precision', 'since_one_month_ago_recall',
#            'since_one_month_ago_pr_auc', 'since_one_month_ago_lift', 'since_one_month_ago_mae',
#            'since_one_month_ago_rmse']

class ScoringMetricsComponent:
    def __init__(self, custom_dictionary=None):
        self.pgdb = self._connect_to_db()
        table = self._load_metrics_table()
        self.metrics_table = self._reformat_metrics_data_for_plotting(table)
        self.metric_options = self._options_for_multi_select_dropdown()
        class_key_words = ['acc', 'roc_auc', 'pr_auc', 'precision', 'recall']
        regress_key_words = ['mae', 'rmse']
        booking_key_words = ['mean_error', 'true_positive']
        self.class_metric_options_dict = self._produce_options_dict(class_key_words)
        print(self.class_metric_options_dict)
        self.regress_metric_options_dict = self._produce_options_dict(regress_key_words)
        self.booking_metric_options_dict = self._produce_options_dict(booking_key_words)
        class_metrics_options = [[option for option in self.metric_options if key_word in option] for key_word in
                                 class_key_words]
        regress_metrics_options = [[option for option in self.metric_options if key_word in option] for key_word in
                                   regress_key_words]
        booking_metrics_options = [[option for option in self.metric_options if key_word in option] for key_word in
                                   booking_key_words]
        self.class_metrics_option_mapping = dict(zip(class_key_words, class_metrics_options))
        print(self.class_metrics_option_mapping)
        self.regress_metrics_option_mapping = dict(zip(regress_key_words, regress_metrics_options))
        self.booking_metrics_option_mapping = dict(zip(booking_key_words, booking_metrics_options))

    def layout(self):
        return html.Div([
            html.H2('LTV model metrics'),
            html.Div([
                html.H3('Select metrics to inspect for the classification model in production'),
                dcc.Dropdown(id='gat-ltv-class-metrics-multiselect', options=self.class_metric_options_dict, value=[self.class_metric_options_dict[0]['value']], multi=True)
            ]),
            dcc.Graph(id='gat-ltv-class-metrics-graph'),
            html.Div([
                html.H3('Select metrics to inspect for the regression model in production'),
                dcc.Dropdown(id='gat-ltv-regress-metrics-multiselect', options=self.regress_metric_options_dict,
                             value=[self.regress_metric_options_dict[0]['value']], multi=True)
            ]),
            dcc.Graph(id='gat-ltv-regress-metrics-graph'),
            html.Div([
                html.H3('Select metrics to inspect for the performance in capturing new bookings in production'),
                dcc.Dropdown(id='gat-ltv-booking-metrics-multiselect', options=self.booking_metric_options_dict,
                             value=[self.booking_metric_options_dict[0]['value']], multi=True)
            ]),
            dcc.Graph(id='gat-ltv-booking-metrics-graph'),
        ], style={'width': '500'})

    def component_callbacks(self, app):
        # the component has three graphs one for classification one for regression and one for live monitoring
        @app.callback(
            Output('gat-ltv-class-metrics-graph', 'figure'),
            [Input('gat-ltv-class-metrics-multiselect', 'value')]
        )
        def update_class_metrics_graph(metrics_to_plot):
            if len(metrics_to_plot) == 0:
                return {}
            else:
                plotting_metrics = []
                for metric in metrics_to_plot:
                    plotting_metrics += self.class_metrics_option_mapping[metric]
                return self._plot_metric_graph(plotting_metrics)

        @app.callback(
            Output('gat-ltv-regress-metrics-graph', 'figure'),
            [Input('gat-ltv-regress-metrics-multiselect', 'value')]
        )
        def update_regress_metrics_graph(metrics_to_plot):
            if len(metrics_to_plot) == 0:
                return {}
            else:
                plotting_metrics = []
                for metric in metrics_to_plot:
                    plotting_metrics += self.regress_metrics_option_mapping[metric]
                return self._plot_metric_graph(plotting_metrics)

        @app.callback(
            Output('gat-ltv-booking-metrics-graph', 'figure'),
            [Input('gat-ltv-booking-metrics-multiselect', 'value')]
        )
        def update_booking_metrics_graph(metrics_to_plot):
            if len(metrics_to_plot) == 0:
                return {}
            else:
                plotting_metrics = []
                for metric in metrics_to_plot:
                    plotting_metrics += self.booking_metrics_option_mapping[metric]
                return self._plot_metric_graph(plotting_metrics)

    @staticmethod
    def _connect_to_db():
        return create_engine(get_postgres_sqlalchemy_uri())

    def _load_metrics_table(self):
        statement = text("""
                   select * from gat.acqusition_ltv_model_metrics order by scoring_date;
                   """)
        with self.pgdb.connect() as con:
            rs = con.execute(statement)
            data = rs.fetchall()
            keys = rs.keys()
            df = pd.DataFrame(data, columns=keys)
        return df

    @staticmethod
    def _reformat_metrics_data_for_plotting(df: DataFrame):
        all_vars = list(df.columns)
        id_vars = 'scoring_date'
        value_vars = all_vars.remove(id_vars)
        df = pd.melt(
            df, id_vars=[id_vars],
            value_vars=value_vars)
        df = df.rename(columns={'variable': 'metric'})
        # reformat the time
        df[id_vars] = pd.to_datetime(df[id_vars],format='%Y%m%d')
        return df

    def _options_for_multi_select_dropdown(self):
        return list(self.metrics_table['metric'].unique())

    @staticmethod
    def _produce_options_dict(metric_options):
        options_dict = []
        for option in metric_options:
            options_dict.append({"label":option, "value": option})
        return options_dict

    def _plot_metric_graph(self, metrics_to_plot):
        filtered_metric_table = self.metrics_table.loc[self.metrics_table.metric.isin(metrics_to_plot), :]
        fig = px.scatter(
            filtered_metric_table,
            x='scoring_date',
            y='value',
            color='metric',
        )
        for d in fig.data:
            d.update(mode='markers+lines')
        return fig



class PostTrainingMonitoringComposite:
    def __init__(self):
        self.scoring_metrics = ScoringMetricsComponent()

    def layout(self):
        return html.Div([
            self.scoring_metrics.layout()
        ], style={'width': '500'})

    def _set_inspection_date(self):
        pass

    def _load_scoring_data(self):
        scoring_data = pd.read_csv(SCORING_FILE)
        return scoring_data

    def _load_registry_data(self):
        pass

    def register_components(self):
        """register components within this compostie so that their callbacks will be registered"""
        if not hasattr(self, '_components'):
            self._components = []
        for k, v in self.__dict__.items():
            if (k != '_components'
                    and v not in self._components):
                self._components.append(v)

    def register_callbacks(self, app):
        self.register_components()
        for comp in self._components:
            comp.component_callbacks(app)




# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
composite = PostTrainingMonitoringComposite()
app.layout = composite.layout()
composite.register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)