import logging.config
import os

import sys

LOGGER = logging.getLogger('ml-logger')
short_time_fmt = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s [logger:%(name)s]', datefmt='%H:%M:%S')
LEVEL = logging.INFO

def add_file_output(logger, filename, level=None):
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    f_handler = logging.FileHandler(filename)
    f_handler.setLevel(level or LEVEL)
    f_handler.setFormatter(short_time_fmt)
    logger.addHandler(f_handler)
    return logger


def console_logger(level=None):
    logger = logging.getLogger()
    logger.setLevel(level or LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level or LEVEL)
    handler.setFormatter(short_time_fmt)
    logger.addHandler(handler)

    return logger


class NegativeFilter(logging.Filter):
    def __init__(self, match_string):
        super().__init__()
        self._match_string = match_string

    def filter(self, record):
        return not (self._match_string in record.getMessage())


simple_logger = console_logger()
