

from flask import jsonify, request
from app.api import main_blueprint
import logging

logger = logging.getLogger(__name__)

@main_blueprint.route('/', methods=['GET'])
def index():
    logger.info(f"request {request.path}")
    return jsonify({'message': 'Muhammed on da code!'})