# flake8: noqa
from flask import Flask


def create_app():
    """Initialize the core application."""
    # Initialize Flask app with '__name__' and 'instance_relative_config=False'
    app = Flask(__name__, instance_relative_config=True)
    # Make sure that we don't keep any cache so images stay fresh after every run
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    with app.app_context():
        from . import routes
        return app
