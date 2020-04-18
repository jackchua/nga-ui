from flask import Blueprint

blueprint = Blueprint(
    'DashMaster_blueprint',
    __name__,
    url_prefix='/DashMaster',
    template_folder='templates',
    static_folder='static'
)
