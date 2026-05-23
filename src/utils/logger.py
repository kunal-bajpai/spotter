import logging
import os
import sys

def setup_logger(name: str = "AI_Squat_Coach", log_file: str = "logs/app.log") -> logging.Logger:
    """Sets up a standardized logger that outputs both to the console and to a log file."""
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    
    # If the logger is already configured, don't add handlers again
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formatting style
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console Handler (Stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Create a default system-wide logger
logger = setup_logger()
