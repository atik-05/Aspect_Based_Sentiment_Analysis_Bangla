import logging
from datetime import datetime

logging.basicConfig(filename='data/log.txt', level=logging.INFO)

logging.info('present word vectors is created at %s...' %datetime.now() )