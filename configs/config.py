import os
import json

# require that a creds.json file has been created / downloaded from a secure
# key store upon deployment
curr_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curr_dir, 'creds.json')) as f:
    creds = json.loads(f.read())

class Config(object):
    # test on a local sqllite database
    SECRET_KEY = b'\x05\xd9.(g\xe1\xbf`\xb1t\xb0n\xeb\xed\x98\xa1'
    SQLALCHEMY_DATABASE_URI = creds['debug']['db']['uri']
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ADMIN = {'username': creds['debug']['admin_user']['username'],
             'email': creds['debug']['admin_user']['email'],
             'password': creds['debug']['admin_user']['password']}

    # THEME SUPPORT
    #  if set then url_for('static', filename='', theme='')
    #  will add the theme name to the static URL:
    #    /static/<DEFAULT_THEME>/filename
    # DEFAULT_THEME = "themes/dark"
    DEFAULT_THEME = None


class ProductionConfig(Config):
    DEBUG = False

    # use the above postgres database
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
        creds['production']['db']['username'],
        creds['production']['db']['password'],
        creds['production']['db']['host'],
        creds['production']['db']['port'],
        'ef'
    )


class DebugConfig(Config):
    DEBUG = True


config_dict = {
    'Production': ProductionConfig,
    'Debug': DebugConfig
}
