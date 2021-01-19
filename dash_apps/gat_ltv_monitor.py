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
import dash_bootstrap_components as dbc
import dash_table
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from explainerdashboard import ClassifierExplainer, RegressionExplainer
import numpy as np
import pickle
import s3fs
import shortuuid
from explainerdashboard.custom import *
from pandas import DataFrame
from .utility import apply_layout_with_auth, load_object, save_object, get_postgres_sqlalchemy_uri, configure_cache

fs = s3fs.S3FileSystem(anon=False)

# # global vars
_URL_BASE = '/dash/gat_ltv_monitor/'
_ACCOUNT_VALUES = ('test')
TIMEOUT = 1800
cache = configure_cache()
TRAINING_DATE = "20201214"

# # Local
# TRAINING_DATA_LOC = 'data/20201214_dataset_v2.csv'  # to be replaced by reading from s3 directly

# Server
TRAINING_DATA_LOC = 's3://ef-nga/dev/workflows/modeling/acquisition_ltv/gat/20201214_dataset_v2.csv'

class MonitoringDashboard:
    def __init__(self, server, title="GAT LTV Acquisition Model", description=None, fluid=False):
        """Initialise the dashboard with passed in params"""
        self.app = Dash(server=server, url_base_pathname=_URL_BASE)
        # figure out how to add cache later
        global cache
        cache.init_app(server)
        data_loader = DataLoader(TRAINING_DATA_LOC, TRAINING_DATE)
        self.tabs = [ClassificationModelComposite(data_loader), RegressionModelComposite(data_loader),
                PostTrainingMonitoringComposite()]
        self.description = description
        self.title = title
        self.fluid = fluid
        # need to register the call backs here
        # for tab in self.tabs:
        #     tab.register_callbacks(self.app)

    def _load_scoring_data(self):
        pass

    def _load_registry_data(self):
        pass

    def _load_upload_data(self):
        pass

    def _load_regression_model(self):
        pass

    def _load_calibration_model(self):
        pass

    def _get_regression_explainer(self):
        pass

    def layout(self):
        """The layout of the app with different tabs"""
        return html.Div([
            dbc.Container([
            dbc.Row([
                    dbc.Col([
                        html.H1(self.title, id='dashboard-title'),
                        dbc.Tooltip(self.description, target='dashboard-title')
                    ], width="auto")], justify="start", style=dict(marginBottom=10)),
            dcc.Tabs(id="tabs", value=self.tabs[0].name,
                     children=[dcc.Tab(label=tab.title, id=tab.name, value=tab.name) for tab in self.tabs]),
            html.Div(
                children=[
                    dcc.Loading(id='tabs-content')
                ],
            )
            ], fluid=self.fluid)
        ], style={'width': '500'})

    def register_callbacks(self, app):
        @app.callback(Output('tabs-content', 'children'),
                      [Input('tabs', 'value')])
        def render_content(tab_value):
            matched_tab = [tab for tab in self.tabs if tab.name==tab_value]
            assert len(matched_tab) == 1
            return matched_tab[0].layout()

    def run(self):
        """Return app.server to be served in via waitress or gunicorn"""
        for tab in self.tabs:
            tab.register_callbacks(self.app)
        self.register_callbacks(self.app)
        apply_layout_with_auth(self.app, self.layout())
        return self.app.server


class DataLoader:
    def __init__(self, training_data_loc, training_data_date):
        self.training_data_loc = training_data_loc
        self.training_data_date = training_data_date

    @cache.memoize(timeout=TIMEOUT)
    def prepare_training_data_for_classification(self):
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        from sklearn.model_selection import train_test_split
        data = self.load_training_data(self.training_data_loc)
        _training_date = datetime.strptime(self.training_data_date, "%Y%m%d")
        three_month_ago_cutoff = _training_date - relativedelta(months=+3)
        old_observed_user = pd.to_datetime(data["first_session_date"]).apply(
            lambda t: t.tz_localize(None)) <= three_month_ago_cutoff
        train_data = data.loc[old_observed_user, :]
        has_booked = (train_data["num_bookings"] > 0).astype(int)
        grandtotal = train_data["GrandTotal"].copy()
        train_data = self.preprocess_training_data(train_data)
        # check and assert there is no more nan in the dataset
        assert train_data.isna().sum().sum() == 0
        # quick sense check: anyone with a positive grandtotal must have booked before and vice versa
        assert (grandtotal[has_booked == 0].sum()) == 0
        assert (has_booked[grandtotal == 0].sum()) == 0
        # assemble the datasets
        y_booked = has_booked
        y_grandtotal = grandtotal
        X = train_data
        X_trainval_booked, X_test_booked, y_trainval_booked, y_test_booked = train_test_split(
            X, y_booked, test_size=0.2, random_state=42
        )
        # more to come with the regression datasets but focus on classification first
        return X_trainval_booked, X_test_booked, y_trainval_booked, y_test_booked, y_grandtotal

    @staticmethod
    def prepare_training_data_for_regression(X_trainval_booked, X_test_booked, y_trainval_booked, y_test_booked, y_grandtotal):
        X_trainval_grandtotal = X_trainval_booked[y_trainval_booked == 1]
        y_trainval_grandtotal = y_grandtotal.loc[
            y_trainval_booked[y_trainval_booked == 1].index
        ]
        X_test_grandtotal = X_test_booked[y_test_booked == 1]
        y_test_grandtotal = y_grandtotal.loc[y_test_booked[y_test_booked == 1].index]
        return X_trainval_grandtotal, y_trainval_grandtotal, X_test_grandtotal, y_test_grandtotal

    @staticmethod
    def load_training_data(training_data_key):
        # latest schema for pandas to reading
        all_vars = [
            "ClientId",
            "first_touch_medium",
            "first_touch_source",
            "first_touch_campaign",
            "last_touch_medium",
            "last_touch_source",
            "last_touch_campaign",
            "last_session_date",
            "first_session_date",
            "last_month_seen",
            "last_week_seen",
            "last_year_seen",
            "TotalTimeonHomePage",
            "TotalTimeonTourPage",
            "NumVisitsonHomePage",
            "NumVisitsonTourPage",
            "TotalTimeonBrowsePage",
            "NumVisitsonBrowsePage",
            "TotalTimeonTravelDealsPage",
            "NumVisitsonTravelDealsPage",
            "TotalTimeOnOtherPage",
            "NumVisitsonOtherPages",
            "TotalTimeonAllPages",
            "AvgTimeonHomePage",
            "AvgTimeonBrowsePage",
            "AvgTimeonTourPage",
            "AvgTimeonTravelDealsPage",
            "AvgTimeonAllPages",
            "AvgTimeonOtherPages",
            "num_sessions",
            "TotalSessionDuration",
            "AvgSessionDuration",
            "bounce_rate",
            "num_sessions_last_week",
            "num_sessions_last_month",
            "num_sessions_last_two_months",
            "num_sessions_last_three_months",
            "num_mobile_sessions",
            "num_tablet_sessions",
            "num_desktop_sessions",
            "num_facebook_sessions",
            "num_cpc_sessions",
            "num_display_sessions",
            "num_organic_sessions",
            "num_referral_sessions",
            "num_social_sessions",
            "num_web_sessions",
            "num_print_sessions",
            "most_frequent_day_of_week",
            "most_frequent_time_of_day",
            "most_frequent_source",
            "avg_session_duration",
            "most_frequent_campaign",
            "most_frequent_medium",
            "num_sessions_first_week",
            "num_sessions_first_month",
            "SalesforceId",
            "daydif",
            "num_bookings",
            "GrandTotal",
            "GoogleClickId",
            "NumBrandCampaigns",
            "hostname",
        ]
        schema = {}
        for var in all_vars:
            # default strings
            schema[var] = str
        data = pd.read_csv(training_data_key, error_bad_lines=False, sep=",", dtype=schema)
        # float points
        num_vars = [
            "last_month_seen",
            "last_week_seen",
            "TotalTimeonHomePage",
            "TotalTimeonTourPage",
            "NumVisitsonHomePage",
            "NumVisitsonTourPage",
            "TotalTimeonBrowsePage",
            "NumVisitsonBrowsePage",
            "TotalTimeonTravelDealsPage",
            "NumVisitsonTravelDealsPage",
            "TotalTimeOnOtherPage",
            "NumVisitsonOtherPages",
            "TotalTimeonAllPages",
            "AvgTimeonHomePage",
            "AvgTimeonBrowsePage",
            "AvgTimeonTourPage",
            "AvgTimeonTravelDealsPage",
            "AvgTimeonAllPages",
            "AvgTimeonOtherPages",
            "num_sessions",
            "TotalSessionDuration",
            "AvgSessionDuration",
            "bounce_rate",
            "num_sessions_last_week",
            "num_sessions_last_month",
            "num_sessions_last_two_months",
            "num_sessions_last_three_months",
            "num_sessions_first_week",
            "num_sessions_first_month",
            "num_mobile_sessions",
            "num_tablet_sessions",
            "num_desktop_sessions",
            "num_facebook_sessions",
            "num_cpc_sessions",
            "num_display_sessions",
            "num_organic_sessions",
            "num_referral_sessions",
            "num_social_sessions",
            "num_web_sessions",
            "num_print_sessions",
            "most_frequent_day_of_week",
            "daydif",
            "num_bookings",
            "GrandTotal",
            "NumBrandCampaigns",
        ]
        num_var_schema = {}
        for var in num_vars:
            num_var_schema[var] = float
        data = data.astype(num_var_schema)
        # exclude people who have 0 session time which is likely a GA bug
        data = data.loc[data.TotalSessionDuration > 0, :]
        print(data.dtypes)
        # preprocess the GoogleClickId
        data["GoogleClickId"] = data["GoogleClickId"].apply(lambda x: np.nan if x == "[]" else x)
        data["SalesforceId"] = data["SalesforceId"].apply(lambda x: np.nan if x == "[]" else x)
        return data

    @staticmethod
    def preprocess_training_data(data):
        """Preprocess training data just like at training time in gat acquisition model"""
        # we scale avg time features by AvgTimeonAllPages
        avg_time_cols = [
            "AvgTimeonHomePage",
            "AvgTimeonBrowsePage",
            "AvgTimeonTourPage",
            "AvgTimeonTravelDealsPage",
            "AvgTimeonOtherPages",
        ]
        for avg_time in avg_time_cols:
            data[avg_time] = data[avg_time] / data["AvgTimeonAllPages"]
        convert_to_str_cols = [
            "last_week_seen",
            "last_month_seen",
            "last_year_seen",
            "most_frequent_day_of_week",
            "most_frequent_time_of_day",
        ]
        for v in convert_to_str_cols:
            data[v] = data[v].astype(str)

        most_freq_mediums = ["cpc", "display", "organic", "email", "referral"]
        medium_features = ["first_touch_medium", "last_touch_medium"]
        for v in medium_features:
            data.loc[:, v] = data.loc[:, v].apply(
                lambda x: "Other" if x not in most_freq_mediums else x
            )

        most_freq_sources = [
            "google",
            "bing",
            "promo",
            "editorial",
            "trigger",
            "cdt",
            "direct",
        ]
        source_features = [
            "most_frequent_source",
            "first_touch_source",
            "last_touch_source",
        ]
        for v in source_features:
            if v in data.columns:
                data.loc[:, v] = data.loc[:, v].fillna("Unknown")
                data.loc[:, v] = data.loc[:, v].apply(
                    lambda x: "Other" if x not in most_freq_sources else x
                )
        columns_to_drop = [
            "num_bookings",
            "first_session_date",
            "last_session_date",
            "GrandTotal",
            "ClientId",
            "GoogleClickId",
            "SalesforceId",
            "CustomerId",
            "daydif",
            "num_sessions_last_week",
            "num_sessions_last_month",
            "num_sessions_last_two_months",
            "num_sessions_last_three_months",
            "num_sessions_first_week",
            "num_sessions_first_month",
            "AvgTimeonAllPages",
        ]
        data = data.drop(columns_to_drop, axis=1)
        return data


def update_params(kwargs, **params):
    """kwargs override params"""
    return dict(params, **kwargs)


class ClassificationModelComposite:
    def __init__(self, data_loader: DataLoader, title="Classfication Model Performance Summary", name=None, hide_selector=True, pos_label=None, cats=True, depth=None, **kwargs):
        self.data_loader = data_loader
        if not hasattr(self, "name") or self.name is None:
            self.name = name or "uuid" + shortuuid.ShortUUID().random(length=5)
        if not hasattr(self, "title") or self.title is None:
            self.title = title or "Custom"
        explainer = self._get_classification_explainer()
        self.summary = ClassifierModelSummaryComponent(explainer, name=self.name + "0",
                                                       hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.precision = PrecisionComponent(explainer, name=self.name + "1",
                                            hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.confusion_matrix = ConfusionMatrixComponent(explainer, name=self.name + "2",
                                                        hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.cumulative_precision = CumulativePrecisionComponent(explainer, name=self.name + "3",
                                                                 hide_selector=hide_selector, pos_label=pos_label,
                                                                 **kwargs)
        self.lift_curve = LiftCurveComponent(explainer, name=self.name + "4",
                                            hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.classification = ClassificationComponent(explainer, name=self.name + "5",
                                                      hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.roc_auc = RocAucComponent(explainer, name=self.name + "6",
                                      hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.pr_auc = PrAucComponent(explainer, name=self.name + "7",
                                    hide_selector=hide_selector, pos_label=pos_label, **kwargs)

        self.cutoff_percentile = CutoffPercentileComponent(explainer, name=self.name + "8",
                                                          hide_selector=hide_selector, pos_label=pos_label, **kwargs)
        self.cutoff_connector = CutoffConnector(self.cutoff_percentile,
                                               [self.summary, self.precision, self.confusion_matrix, self.lift_curve,
                                                self.cumulative_precision, self.classification, self.roc_auc,
                                                self.pr_auc])
        self.shap_summary = ShapSummaryComponent(
            explainer, name=self.name + "9",
            **update_params(kwargs, hide_selector=hide_selector, depth=depth, cats=cats))
        self.shap_dependence = ShapDependenceComponent(
            explainer, name=self.name + "10",
            hide_selector=hide_selector, cats=cats,
            **update_params(kwargs, hide_cats=True)
        )
        self.connector = ShapSummaryDependenceConnector(
            self.shap_summary, self.shap_dependence)

    @staticmethod
    def _load_classification_model():
        classification_model = CatBoostClassifier()
        # # Local
        # MODEL_PATH = "model/20201214_classification_has_booked"
        # classification_model.load_model(MODEL_PATH)
        # Server
        MODEL_PATH = "s3://ef-nga/prod/workflows/modeling/acquisition_ltv/gat/ltv-experiment/20201214_classification_has_booked"
        classification_model.load_model(stream=fs.open(MODEL_PATH, 'rb'))
        return classification_model

    def _get_classification_explainer(self):
        classification_model = self._load_classification_model()
        _, x_test_booked, _, y_test_booked, _ = self.data_loader.prepare_training_data_for_classification()
        explainer = ClassifierExplainer(classification_model, x_test_booked, y_test_booked,
                                        labels=['Not booked', 'booked'],  # defaults to ['0', '1', etc]
                                        index_name="Browser row number",  # defaults to X.index.name
                                        )
        return explainer

    def layout(self):
        # wrapped by dbc card
        return html.Div([
            dbc.Row([
                    dbc.Col([
                        html.H2('Model Performance:')])]),
            dbc.Row([
                    dbc.Col([
                        self.cutoff_percentile.layout(),
                    ])], style=dict(marginTop=25, marginBottom=25)),
            dbc.CardDeck([
                self.summary.layout(),
                self.confusion_matrix.layout(),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.precision.layout(),
                self.classification.layout()
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.roc_auc.layout(),
                self.pr_auc.layout(),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.lift_curve.layout(),
                self.cumulative_precision.layout(),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.shap_summary.layout(),
                self.shap_dependence.layout(),
            ], style=dict(marginTop=25)),
        ])

    def register_components(self):
        """register components within this compostie so that their callbacks will be registered"""
        if not hasattr(self, '_components'):
            self._components = []
        for k, v in self.__dict__.items():
            if (k != '_components'
                    and isinstance(v, ExplainerComponent)
                    and v not in self._components):
                self._components.append(v)

    def register_callbacks(self, app):
        self.register_components()
        for comp in self._components:
            comp.component_callbacks(app)


class RegressionModelComposite:
    def __init__(self, data_loader: DataLoader, title="Regression Model Performance Summary", name=None, pred_or_actual="vs_pred", residuals='difference', logs=False, cats=True, hide_selector=True, depth=None, **kwargs):
        self.data_loader = data_loader
        if not hasattr(self, "name") or self.name is None:
            self.name = name or "uuid" + shortuuid.ShortUUID().random(length=5)
        if not hasattr(self, "title") or self.title is None:
            self.title = title or "Custom"
        explainer = self._get_regression_explainer()
        assert pred_or_actual in ['vs_actual', 'vs_pred'], \
            "pred_or_actual should be 'vs_actual' or 'vs_pred'!"
        self.modelsummary = RegressionModelSummaryComponent(explainer,
                                                            name=self.name + "0", **kwargs)
        self.preds_vs_actual = PredictedVsActualComponent(explainer, name=self.name + "1",
                                                          logs=logs, **kwargs)
        self.residuals = ResidualsComponent(explainer, name=self.name + "2",
                                            pred_or_actual=pred_or_actual, residuals=residuals, **kwargs)
        self.reg_vs_col = RegressionVsColComponent(explainer, name=self.name + "3",
                                                   logs=logs, **kwargs)
        self.shap_summary = ShapSummaryComponent(
            explainer, name=self.name + "4",
            **update_params(kwargs, hide_selector=hide_selector, depth=depth, cats=cats))
        self.shap_dependence = ShapDependenceComponent(
            explainer, name=self.name + "5",
            hide_selector=hide_selector, cats=cats,
            **update_params(kwargs, hide_cats=True)
        )
        self.connector = ShapSummaryDependenceConnector(
            self.shap_summary, self.shap_dependence)

    def layout(self):
        # wrapped by dbc card
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2('Model Performance:')])]),
            dbc.CardDeck([
                self.modelsummary.layout(),
                self.preds_vs_actual.layout(),
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.residuals.layout(),
                self.reg_vs_col.layout()
            ], style=dict(marginBottom=25)),
            dbc.CardDeck([
                self.shap_summary.layout(),
                self.shap_dependence.layout(),
            ], style=dict(marginTop=25)),
        ])

    @staticmethod
    def _load_regression_model():
        regression_model = CatBoostRegressor()
        # # Local
        # MODEL_PATH = "model/20201214_regression_grandtotal"
        # regression_model.load_model(MODEL_PATH)
        # Server
        MODEL_PATH = "s3://ef-nga/prod/workflows/modeling/acquisition_ltv/gat/ltv-experiment/20201214_regression_grandtotal"
        regression_model.load_model(stream=fs.open(MODEL_PATH, 'rb'))
        return regression_model

    def _get_regression_explainer(self):
        regression_model = self._load_regression_model()
        X_trainval_booked, X_test_booked, y_trainval_booked, y_test_booked, y_grandtotal = self.data_loader.prepare_training_data_for_classification()
        _, _, X_test_grandtotal, y_test_grandtotal = self.data_loader.prepare_training_data_for_regression(X_trainval_booked, X_test_booked, y_trainval_booked, y_test_booked, y_grandtotal)
        explainer = RegressionExplainer(regression_model, X_test_grandtotal, y_test_grandtotal)
        return explainer

    def register_components(self):
        """register components within this compostie so that their callbacks will be registered"""
        if not hasattr(self, '_components'):
            self._components = []
        for k, v in self.__dict__.items():
            if (k != '_components'
                    and isinstance(v, ExplainerComponent)
                    and v not in self._components):
                self._components.append(v)

    def register_callbacks(self, app):
        self.register_components()
        for comp in self._components:
            comp.component_callbacks(app)


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
        self.regress_metric_options_dict = self._produce_options_dict(regress_key_words)
        self.booking_metric_options_dict = self._produce_options_dict(booking_key_words)
        class_metrics_options = [[option for option in self.metric_options if key_word in option] for key_word in
                                 class_key_words]
        regress_metrics_options = [[option for option in self.metric_options if key_word in option] for key_word in
                                   regress_key_words]
        booking_metrics_options = [[option for option in self.metric_options if key_word in option] for key_word in
                                   booking_key_words]
        self.class_metrics_option_mapping = dict(zip(class_key_words, class_metrics_options))
        self.regress_metrics_option_mapping = dict(zip(regress_key_words, regress_metrics_options))
        self.booking_metrics_option_mapping = dict(zip(booking_key_words, booking_metrics_options))

    def layout(self):
        return html.Div([
            dcc.Interval(interval=24 * 3600 * 1000, id="interval"),
            html.Div(id='hidden-div', style = {'display':'none'}),
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
        # pull the latest table on DB on a schedule
        @app.callback(
            Output('hidden-div', 'children'),
            [Input("interval", "n_intervals")]
        )
        def update_metrics_table(_):
            # update the DB from client side (might get out of hand if the table grows too big
            table = self._load_metrics_table()
            self.metrics_table = self._reformat_metrics_data_for_plotting(table)
            return None

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
    def __init__(self, title='Production metrics', name=None):
        self.scoring_metrics = ScoringMetricsComponent()
        if not hasattr(self, "name") or self.name is None:
            self.name = name or "uuid" + shortuuid.ShortUUID().random(length=5)
        if not hasattr(self, "title") or self.title is None:
            self.title = title or "Custom"

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
                    and v not in self._components
                    and isinstance(v, ScoringMetricsComponent)):
                self._components.append(v)

    def register_callbacks(self, app):
        self.register_components()
        for comp in self._components:
            comp.component_callbacks(app)



