# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides the logging capability"""
import contextlib
import inspect
import json
import logging.handlers
import os
import sys
import threading
import uuid
from datetime import datetime
from datetime import timezone

import humanize

ENVIRONMENT_VAR_NAME = "INTELLIGENCE_PATH"
# check if environment variable is set
env_var = os.getenv(ENVIRONMENT_VAR_NAME) or os.getenv(ENVIRONMENT_VAR_NAME.lower())
if env_var:
    # python gets environment variable with quotes
    env_var = env_var.replace('"', "").replace("'", "").strip()
else:
    print(
        f"Environment variable {ENVIRONMENT_VAR_NAME} not set. Set it to use the application."
    )
    sys.exit(1)

# DEFINE LOGGING CONSTANTS
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_LOG_NAME = "app_Log"
DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S %z"
DEFAULT_FILE_LOG_DIR = os.path.join(env_var, "konaai_logs")
os.makedirs(DEFAULT_FILE_LOG_DIR, exist_ok=True)


class AnsiColorCode:
    """ANSI color codes for terminal colors"""

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"


# Custom formatter with color support
class ColorFormatter(logging.Formatter):
    """Custom formatter with color support"""

    COLOR_MAP = {
        logging.DEBUG: AnsiColorCode.BLUE,
        logging.INFO: AnsiColorCode.WHITE,
        logging.WARNING: AnsiColorCode.YELLOW,
        logging.ERROR: AnsiColorCode.RED,
        logging.CRITICAL: AnsiColorCode.MAGENTA,
        45: AnsiColorCode.RED,
        60: AnsiColorCode.GREEN,
    }

    def format(self, record):
        level_color = self.COLOR_MAP.get(record.levelno, AnsiColorCode.WHITE)

        try:
            formatted_message = super().format(record)
        except Exception as e:
            formatted_message = (
                f"[Error formatting log message: {e}] {record.getMessage()}"
            )

        if record.exc_info:
            try:
                exception_text = self.formatException(record.exc_info)
                formatted_message += f"\n{exception_text}"
            except Exception as e:
                formatted_message += f"\nError formatting exception: {e}"

        return f"{level_color}{formatted_message}{AnsiColorCode.RESET}"


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        """Format log record as JSON object."""
        try:
            correlation_id = str(
                uuid.uuid5(uuid.NAMESPACE_OID, f"{threading.current_thread().ident}")
            )
            log_record = {
                "timestamp": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "filename": record.filename,
                "function": record.funcName,
                "line": record.lineno,
                "correlation_id": correlation_id,
            }

            if record.exc_info:
                log_record["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "traceback": self.formatException(record.exc_info),
                }

            # Add custom attributes
            for key, value in record.__dict__.items():
                if key.startswith("custom_") and key not in log_record:
                    log_record[key] = value

            return json.dumps(log_record, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "ERROR",
                    "message": f"Error formatting log record: {e}",
                }
            )


class CustomLogger:
    """Flexible logging utility with JSON file logging and colored console output"""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        log_level=DEFAULT_LOG_LEVEL,
        log_name=DEFAULT_LOG_NAME,
        time_format=DEFAULT_TIME_FORMAT,
        log_dir=DEFAULT_FILE_LOG_DIR,
        enable_stream=True,
    ):
        self.enable_stream = enable_stream
        self.log_level = log_level
        self.log_name = log_name
        self.time_format = time_format
        self.file_log_dir = log_dir
        self._context_data = {}

        # Initialize handler attributes
        self.file_handler = None
        self.stream_handler = None

        # Create unique logger to avoid conflicts
        self.logger = logging.getLogger(f"{self.log_name}_{id(self)}")

        # Add custom log levels
        logging.addLevelName(45, "EXCEPTION")
        logging.addLevelName(60, "SUCCESS")

        self.logger.setLevel(self.log_level)
        self.logger.handlers.clear()
        self.logger.propagate = False

        if self.enable_stream:
            self.create_stream_handler()
        self.create_file_handler()

    def create_file_handler(self):
        """Creates file handler with error handling"""
        try:
            if self.file_handler:
                self.logger.removeHandler(self.file_handler)
                self.file_handler.close()

            os.makedirs(self.file_log_dir, exist_ok=True)
            file_name = f"{self.log_name}.json"
            self.log_file_path = os.path.join(self.file_log_dir, file_name)

            self.file_handler = logging.handlers.TimedRotatingFileHandler(
                self.log_file_path,
                when="midnight",
                interval=1,
                backupCount=90,
                encoding="utf-8",
                utc=False,
            )
            self.file_handler.setLevel(self.log_level)
            self.file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(self.file_handler)
        except Exception as e:
            print(f"Warning: Failed to create file handler: {e}")

    def create_stream_handler(self):
        """Creates colored console stream handler"""
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(self.log_level)
        self.stream_handler.setFormatter(
            ColorFormatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt=self.time_format,
            )
        )
        self.logger.addHandler(self.stream_handler)

    def set_level(self, level):
        """Sets log level for all handlers"""
        self.log_level = level
        self.logger.setLevel(level)
        if self.stream_handler:
            self.stream_handler.setLevel(level)
        if self.file_handler:
            self.file_handler.setLevel(level)

    def create_record(self, msg, level, debug=False, exception=None, **kwargs):
        """Creates log record with context"""
        stack = inspect.stack()
        record = logging.LogRecord(
            name=self.log_name,
            level=level,
            pathname=stack[2][1],
            lineno=stack[2][2],
            msg=msg,
            args=None,
            exc_info=exception if debug else None,
            func=stack[2][3],
        )

        # Add context and kwargs
        for key, value in {**self._context_data, **kwargs}.items():
            setattr(record, f"custom_{key}", value)

        return record

    def add_context(self, **kwargs):
        """Add contextual information to all subsequent logs"""
        self._context_data.update(kwargs)

    def clear_context(self):
        """Clear all contextual information"""
        self._context_data.clear()

    def remove_context(self, *keys):
        """Remove specific context keys"""
        for key in keys:
            self._context_data.pop(key, None)

    @contextlib.contextmanager
    def context(self, **kwargs):
        """Context manager for temporary logging context"""
        original_context = self._context_data.copy()
        self._context_data.update(kwargs)
        try:
            yield
        finally:
            self._context_data = original_context

    def info(self, msg, **kwargs):
        """Info with optional context"""
        record = self.create_record(msg, logging.INFO, **kwargs)
        self.logger.handle(record)

    def warning(self, msg, **kwargs):
        """Warning with optional context"""
        record = self.create_record(msg=msg, level=logging.WARNING, **kwargs)
        self.logger.handle(record)

    def error(self, msg, traceback: bool = True, **kwargs):
        """Error with enhanced exception handling"""
        record = self.create_record(
            msg=msg,
            level=logging.ERROR,
            debug=True,
            exception=(
                sys.exc_info() if sys.exc_info()[0] is not None and traceback else None
            ),
            **kwargs,
        )
        self.logger.handle(record)

    def debug(self, msg, **kwargs):
        """Debug with optional context"""
        record = self.create_record(msg, logging.DEBUG, **kwargs)
        self.logger.handle(record)

    def critical(self, msg, **kwargs):
        """Critical with optional context"""
        record = self.create_record(msg, logging.CRITICAL, **kwargs)
        self.logger.handle(record)

    def perflog(self, duration, size, operation, **kwargs):
        """Log performance metrics"""
        msg = f"Performance: {operation} - {humanize.naturaldelta(duration)} ({duration:.3f}s), {humanize.naturalsize(size)}"
        record = self.create_record(
            msg,
            logging.INFO,
            duration=duration,
            size=size,
            operation=operation,
            **kwargs,
        )
        self.logger.handle(record)

    def exception(self, msg, **kwargs):
        """Exception with enhanced context"""
        record = self.create_record(
            level=45, msg=msg, debug=True, exception=sys.exc_info(), **kwargs
        )
        self.logger.handle(record)

    def success(self, msg, **kwargs):
        """Success with optional context"""
        record = self.create_record(msg, 60, **kwargs)
        self.logger.handle(record)

    def close(self):
        """Properly close all handlers"""
        if self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
        if self.stream_handler:
            self.logger.removeHandler(self.stream_handler)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


app_logger = CustomLogger(log_name="KonaAIML")
