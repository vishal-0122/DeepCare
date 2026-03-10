import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger instance with the given name.
    Logs are saved in logs/app_<date>.log and also printed to console.
    Log files rotate daily at midnight and keep 7 days of history.

    Args:
        name (str): Name of the logger (usually __name__ of the module).
        level (int): Logging level (default = INFO). Can be DEBUG, WARNING, ERROR, CRITICAL.

    Returns:
        logging.Logger: Configured logger instance.
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Log file base name with unique identifier
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y-%m-%d')}.log")

        # Logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Prevent duplicate handlers
        if not logger.handlers:
            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # File Handler with daily rotation (keeps 7 days of logs)
            file_handler = TimedRotatingFileHandler(
                log_file, when="midnight", backupCount=7, encoding="utf-8"
            )
            file_handler.setLevel(level)

            # Format for logs
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger
    except Exception as e:
        print(f"Failed to set up logger: {e}")
        raise
