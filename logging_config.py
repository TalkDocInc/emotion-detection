import os
import logging
from logging.handlers import RotatingFileHandler

# Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Create rotating file handler
log_file = os.path.join(log_dir, 'app.log')
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
file_handler.setLevel(logging.INFO)

# Formatter for file logs
formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
file_handler.setFormatter(formatter)

# Attach handler to root logger
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
