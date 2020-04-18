from . import blueprint
from flask import render_template
from flask_login import login_required
from dash_apps import google_ad_budget_optimizer

@blueprint.route('/google_ad_budget_optimizer')
@login_required
def app1_template():
    return render_template('google_ad_budget_optimizer.html', dash_url = google_ad_budget_optimizer._URL_BASE)