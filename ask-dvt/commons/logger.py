import logging
import logging.handlers
import os


class CustomLogger:
    """
    A custom logger class that encapsulates the setup of a Python logger.
    This version only logs to stdout and stderr.
    """

    def __init__(
        self,
        level=logging.INFO,
        log_format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    ):
        """
        Initializes the custom logger.

        Args:
            level (int):  Minimum logging level (e.g., logging.DEBUG, logging.INFO).
            log_format (str): Format string for log messages.
        """
        self.level = level
        self.log_format = log_format
        self._configure_logger()  # Call the setup method

    def _configure_logger(self):
        """
        Configures the logger with handlers and formatter.  This is called
        internally by the __init__ method.  This version only sets up a
        StreamHandler (for stdout/stderr).
        """
        # 1. Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.level)
        self.logger.propagate = False  # Prevent duplicate logging

        # 2. Create a formatter
        formatter = logging.Formatter(self.log_format)

        # 3. Create and configure a StreamHandler for console output
        # Check if a StreamHandler already exists
        if not any(
            isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers
        ):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logging.Logger object.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    def log_debug(self, message: str, *args, **kwargs):
        """Log a message at the DEBUG level."""
        self.logger.debug(message, *args, **kwargs)

    def log_info(self, message: str, *args, **kwargs):
        """Log a message at the INFO level."""
        self.logger.info(message, *args, **kwargs)

    def log_warning(self, message: str, *args, **kwargs):
        """Log a message at the WARNING level."""
        self.logger.warning(message, *args, **kwargs)

    def log_error(self, message: str, *args, **kwargs):
        """Log a message at the ERROR level."""
        self.logger.error(message, *args, **kwargs)

    def log_critical(self, message: str, *args, **kwargs):
        """Log a message at the CRITICAL level."""
        self.logger.critical(message, *args, **kwargs)

    def log_exception(self, message: str, exc_info=True, *args, **kwargs):
        """Log a message at the ERROR level, including exception info."""
        self.logger.exception(message, exc_info=exc_info, *args, **kwargs)