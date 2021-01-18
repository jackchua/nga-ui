from . import blueprint
from flask import render_template
from flask_login import login_required
from dash_apps import google_ads_budget_optimizer, google_ads_ltv_optimizer, google_ads_causal_inference, gat_ltv_monitor

@blueprint.route('/google_ads_budget_optimizer')
@login_required
def google_ads_budget_optimizer_template():
    return render_template(
        'google_ads_budget_optimizer.html',
        dash_url = google_ads_budget_optimizer._URL_BASE
    )

@blueprint.route('/google_ads_ltv_optimizer')
@login_required
def google_ads_ltv_optimizer_template():
    return render_template(
        'google_ads_ltv_optimizer.html',
        dash_url = google_ads_ltv_optimizer._URL_BASE
    )

@blueprint.route('/google_ads_causal_inference')
@login_required
def google_ads_causal_inference_template():
    return render_template(
        'google_ads_causal_inference.html',
        dash_url = google_ads_causal_inference._URL_BASE
    )

@blueprint.route('/gat_ltv_monitor')
@login_required
def gat_ltv_monitor_template():
    return render_template(
        'gat_ltv_monitor.html',
        dash_url = gat_ltv_monitor._URL_BASE
    )