from flask import request, jsonify
from app.api import main_blueprint
import logging


logger = logging.getLogger(__name__)

@main_blueprint.errorhandler(404)
def not_found(error):
    """ Handle  404 error"""
    logger.warning(f"404 error => {request.path} \n {error}")
    return (
        jsonify({
            "message" :"Page Not Found",
        })
    , 404)

@main_blueprint.errorhandler(500)
def server_error(error):
    """ Handle  500 error"""
    logger.warning(f"500 error => {request.path} \n {error}")
    return (
        jsonify({
            "message" :"Internal Server Error",
        }, 500)
    )

@main_blueprint.errorhandler(400)
def bad_request(error):
    """ Handle  400 error """
    logger.warning(f"400 error => {request.path} \n {error}")
    return (
        jsonify({
            "message" :"Bad Request",
            "error" : str(error)
        })
    )

@main_blueprint.errorhandler(405)
def server_error(error):
    """ Handle  501 error"""
    logger.warning(f"405 error => {request.path} \n {error}")
    return (
        jsonify({
            "message" :"Method Not Allowed",
        }, 405)
    )
@main_blueprint.errorhandler(Exception)
def exception_handler(error):
    """ Handle  Exception """
    logger.warning(f"Exception error => {request.path} \n {error}")
    return (
        jsonify({
            "message" :"Internal Server Error --> Exception",
        })
    )