from . import blueprint
from flask import render_template
from flask_login import login_required
from dash_apps import google_ad_budget_optimizer, Dash_App2

@blueprint.route('/app1')
@login_required
def app1_template():
    return render_template('app1.html', dash_url = google_ad_budget_optimizer.url_base)

@blueprint.route('/app2')
@login_required
def app2_template():
    return render_template('app2.html', dash_url = Dash_App2.url_base)