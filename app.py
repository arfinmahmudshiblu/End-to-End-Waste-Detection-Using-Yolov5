# from wasteDetection.logger import logging

# logging.info("Welcome to my custom log")

from wasteDetection.logger import logging
from wasteDetection.exception import AppException
import sys

try:
    r=3/0
except Exception as e:
    logging.info(e)
    raise AppException(e, sys) 