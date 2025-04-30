import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-development'
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'model.pkl')
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'INFO'

class TestingConfig(Config):
    TESTING = True
    LOG_LEVEL = 'DEBUG'