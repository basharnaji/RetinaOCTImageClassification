# flake8: noqa

from flask import Flask


def create_app():
    """Initialize the core application."""
    # Initialize Flask app with '__name__' and 'instance_relative_config=False'
    app = Flask(__name__, instance_relative_config=False)

    with app.app_context():
        from . import routes
        return app
