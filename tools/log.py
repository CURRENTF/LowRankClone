import logging  
import os  
from datetime import datetime  


def get_time_str():
    """
    Get the current time and return it in the format 'yymmdd-hhmmss'.

    :return: String of the current time in the format 'yymmdd-hhmmss'
    """
    return datetime.now().strftime("%y%m%d-%H%M%S")


# Create a logger and configure it
def create_logger(
    name, log_path, terminal_level=logging.INFO, file_level=logging.DEBUG
):
    """
    Create a logger that outputs logs to both the console and a log file.

    :param name: The name of the logger
    :param log_path: The path to the log file
    :param terminal_level: Log level for console output, default is INFO
    :param file_level: Log level for file output, default is DEBUG
    :return: The configured logger object
    """
    # Get a named logger
    logger = logging.getLogger(name)

    # If handlers already exist, return the existing logger to avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    # Set the minimum log level for the logger (set to DEBUG here, meaning all levels are handled)
    logger.setLevel(logging.DEBUG)

    # Create the log file directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(file_level)  # Set log level for file
    file_handler.setFormatter(formatter)  # Set log format

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(terminal_level)  # Set log level for console
    console_handler.setFormatter(formatter)  # Set log format

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Get an existing logger
def get_logger(name):
    """
    Get an already created logger object.

    :param name: The name of the logger
    :return: The logger object
    """
    return logging.getLogger(name)


main_logger = create_logger("main", f"./logs/{get_time_str()}.log")