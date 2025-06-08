import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(log_filename: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Sets up a structured logger with file and console output."""
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(log_filename)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        return logger

    # Formatter: [Timestamp] LEVEL - Message
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_filepath, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
