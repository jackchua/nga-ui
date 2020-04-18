from flask_migrate import Migrate
from configs.config import config_dict
from app import create_app, db
import os
import sys

get_config_mode = os.environ.get('NGA_UI_CONFIG_MODE', 'Debug')

try:
    config_mode = config_dict[get_config_mode.capitalize()]
except KeyError:
    sys.exit('Error: Invalid config environment variable entry.')

app = create_app(config_mode)
Migrate(app, db)

if __name__ == "__main__":
    app.run(debug = True)