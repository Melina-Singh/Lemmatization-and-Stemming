import logging
import os
from datetime import datetime

def setup_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Name of the logger (usually __name__ of the calling module).
        log_file (str): Path to the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, filter by handler

    # Prevent duplicate handlers if logger is reused
    if not logger.handlers:
        # Create file handler (writes to app.log)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Create console handler (prints to terminal)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Show INFO and above in console
        console_formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Example usage (for testing logger.py standalone)
if __name__ == "__main__":
    logger = setup_logger(__name__)
    logger.debug("Debug message for detailed tracking")
    logger.info("Info message for general steps")
    logger.warning("Warning message for potential issues")
    logger.error("Error message for failures")