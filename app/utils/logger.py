import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(app):
    """Configure application logging"""
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))

    # Ensure log directory exists
    log_dir = app.config.get('LOG_DIR', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'app.log')

    # Configure handler
    handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=10)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    # Add handler to app logger
    app.logger.addHandler(handler)
    app.logger.setLevel(log_level)

    # Configure root logger for other modules
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Also configure console logging
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    app.logger.info("Logging setup completed")