import logging
import sys

def setup_logger(name="AcademicAgent"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler = sys.stdout
        handler = logging.StreamHandler(console_handler)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()