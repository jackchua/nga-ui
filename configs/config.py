import json

# require that a creds.json file has been created / downloaded from a secure
# key store upon deployment
with open('creds.json') as f:
    creds = json.loads(f.read())

class Config(object):
    SECRET_KEY = b'\x05\xd9.(g\xe1\xbf`\xb1t\xb0n\xeb\xed\x98\xa1'
    SQLALCHEMY_DATABASE_URI = creds['debug']['db']['uri']
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ADMIN = {'username': creds['debug']['db']['username'],
             'email': creds['debug']['db']['email'],
             'password': creds['debug']['db']['password']}

    # THEME SUPPORT
    #  if set then url_for('static', filename='', theme='')
    #  will add the theme name to the static URL:
    #    /static/<DEFAULT_THEME>/filename
    # DEFAULT_THEME = "themes/dark"
    DEFAULT_THEME = None


class ProductionConfig(Config):
    DEBUG = False

    # PostgreSQL database
    SQLALCHEMY_DATABASE_URI = 'postgresql://{}:{}@{}:{}/{}'.format(
        creds['production']['db']['user'],
        creds['production']['db']['password'],
        creds['production']['db']['host'],
        creds['production']['db']['port'],
        creds['production']['db']['database']
    )


class DebugConfig(Config):
    DEBUG = True


config_dict = {
    'Production': ProductionConfig,
    'Debug': DebugConfig
}
