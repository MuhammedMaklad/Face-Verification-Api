from flask import Blueprint

main_blueprint = Blueprint('main', __name__)

# Import routes to register them with the blueprint
from app.api import routes, error_handlers
