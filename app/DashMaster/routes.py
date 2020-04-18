from . import blueprint
from flask import render_template
from flask_login import login_required
from dash_apps import google_ads_budget_optimizer

@blueprint.route('/google_ads_budget_optimizer')
@login_required
def google_ads_budget_optimizer_template():
    return render_template(
        'google_ads_budget_optimizer.html',
        dash_url = google_ads_budget_optimizer._URL_BASE
    )