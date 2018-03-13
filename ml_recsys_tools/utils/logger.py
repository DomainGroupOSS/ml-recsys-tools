import logging.config
import os

"""
Logging configuration
"""
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '[%(asctime)s][%(levelname)s](%(filename)s:%(lineno)s).%(funcName)s - %(message)s'
        }
    },
    'handlers': {
        'stream_handler': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        }
    },
    'loggers': {
        'image-floorplan': {
            'level': 'DEBUG',
            'handlers': ['stream_handler', ]
        },
        'default': {
            'level': 'DEBUG',
            'handlers': ['stream_handler', ]
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
LOGGER = logging.getLogger('ml-logger')
short_time_fmt = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s', datefmt='%H:%M')

def add_file_output(logger, filename, level=logging.DEBUG):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.mkdir(dir)
    f_handler = logging.FileHandler(filename)
    f_handler.setLevel(level)
    f_handler.setFormatter(short_time_fmt)
    logger.addHandler(f_handler)
    return logger

def console_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(short_time_fmt)
    logger.addHandler(handler)

    return logger


simple_logger = console_logger()

