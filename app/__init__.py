from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS
import os
load_dotenv()

def create_app():
    app = Flask(__name__)

    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
    app.config.from_object('app.config')

    UPLOAD_FOLDER = 'uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    # initialize logger
    from app.utils.logger import setup_logging
    setup_logging(app)

    # Register blueprints
    from app.api import main_blueprint
    app.register_blueprint(main_blueprint)

    return app