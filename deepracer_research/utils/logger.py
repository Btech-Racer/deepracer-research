import inspect
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter that includes caller information and structured context

    Parameters
    ----------
        record: logging.LogRecord
            The log record to format

    Returns
    -------
        str: Formatted log message
    """

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f %Z")

        if not hasattr(record, "caller_info"):
            record.caller_info = self._get_caller_info()

        log_message = record.getMessage()

        if hasattr(record, "extra") and record.extra:
            formatted_items = []
            for k, v in record.extra.items():
                if v is not None:
                    if isinstance(v, (dict, list)):
                        v = json.dumps(v, default=str)
                    formatted_items.append(f"{k}: {v}")

            if formatted_items:
                context_str = " | ".join(formatted_items)
                log_message += f" | Context: {context_str}"

        return f"[{record.levelname}] {timestamp} | {record.caller_info} | {log_message}"

    def _get_caller_info(self, depth: int = 8):
        """
        Determine the caller's filename and function name

        Parameters
        ----------
            depth: int
                The number of frames to go back in the stack to find the caller info

        Returns
            str: Caller information in the format "filename:function_name"
        """
        frame = inspect.currentframe()
        try:
            for _ in range(depth):
                if frame is None:
                    break
                frame = frame.f_back

            if frame:
                function_name = frame.f_code.co_name
                filename = os.path.basename(frame.f_code.co_filename)
                return f"{filename}:{function_name}"
            return "unknown:unknown"
        finally:
            del frame


class ExtraContextFilter(logging.Filter):
    """
    Filter that ensures every record has an extra context dict

    This is useful for ensuring that all log records have a consistent structure
    and can be used to add additional context information.

    Parameters
    ----------
        record: logging.LogRecord
            The log record to filter
    Returns
    -------
        bool: Always returns True, allowing the record to be processed
    """

    def filter(self, record):
        if not hasattr(record, "extra"):
            record.extra = {}
        return True


def setup_logger(log_level: Optional[Union[int, str]] = None) -> logging.Logger:
    """
    Setup the root logger and deepracer_research logger with the specified level

    Parameters
    ----------
        log_level: str or int, optional
            The logging level to set. If None, defaults to INFO.
            Can be a string like 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.

    Returns
    -------
        logging.Logger: Configured logger instance

    Raises
    ------
        ValueError: If the log_level is not a valid logging level
    """
    if log_level is None:
        level = logging.INFO
    else:
        level = log_level

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = CustomFormatter()
    console_handler.setFormatter(formatter)

    context_filter = ExtraContextFilter()
    console_handler.addFilter(context_filter)

    root_logger.addHandler(console_handler)

    app_logger = logging.getLogger("deepracer_research")
    app_logger.setLevel(level)

    return app_logger


def update_log_level(level: Union[str, int]) -> logging.Logger:
    """
    Update the log level after configuration is fully loaded

    Parameters
    ----------
        level: str or int
            The logging level to set. Can be a string like 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
            If an integer, it should be a valid logging level constant.

    Returns
    -------
        logging.Logger: The updated logger instance

    Raises
    ------
        ValueError: If the level is not a valid logging level
    """
    global logger
    logger = setup_logger(level)
    return logger


def _log_factory(level: int) -> Callable:
    """
    Create a logging function for the specified level

    Parameters
    ----------
        level: int
            The logging level to create a function for (e.g., logging.DEBUG, logging.INFO, etc.)

    Returns
    -------
        Callable: A logging function that takes a message and optional extra context
        and logs it at the specified level.

    Raises
    ------
        ValueError: If the level is not a valid logging level
    """

    @wraps(getattr(logging, logging.getLevelName(level).lower()))
    def log_function(message: str, extra: Optional[Dict[str, Any]] = None, caller_info: Optional[str] = None):
        extra_data = extra or {}

        record = logging.LogRecord(
            name="deepracer_research", level=level, pathname="", lineno=0, msg=message, args=(), exc_info=None
        )
        record.extra = extra_data
        if caller_info:
            record.caller_info = caller_info
        logger.handle(record)

    return log_function


def log_execution(func: Callable) -> Callable:
    """
    Decorator to log function execution time and details.
    Handles both synchronous and asynchronous functions.

    Parameters
    ----------
        func: Function to decorate

    Returns
    -------
        Decorated function
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()

        method_name = func.__name__
        class_name = args[0].__class__.__name__ if args and hasattr(args[0], "__class__") else "None"
        context = {"method": method_name, "class": class_name}

        try:
            info(f"Starting {class_name}.{method_name}", context)
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            context["elapsed_time_ms"] = round(elapsed_time * 1000, 2)
            info(f"Completed {class_name}.{method_name}", context)

            return result
        except Exception as e:
            elapsed_time = time.time() - start_time

            context.update(
                {
                    "elapsed_time_ms": round(elapsed_time * 1000, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_details": traceback.format_exc(),
                }
            )

            error(f"Exception in {class_name}.{method_name}", context)

            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()

        method_name = func.__name__
        class_name = args[0].__class__.__name__ if args and hasattr(args[0], "__class__") else "None"
        context = {"method": method_name, "class": class_name}

        try:
            info(f"Starting {class_name}.{method_name}", context)
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            context["elapsed_time_ms"] = round(elapsed_time * 1000, 2)
            info(f"Completed {class_name}.{method_name}", context)

            return result
        except Exception as e:
            elapsed_time = time.time() - start_time

            context.update(
                {
                    "elapsed_time_ms": round(elapsed_time * 1000, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_details": traceback.format_exc(),
                }
            )

            error(f"Exception in {class_name}.{method_name}", context)

            raise

    if inspect.iscoroutinefunction(func):
        return async_wrapper

    return sync_wrapper


def exception(
    message: str, exc: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None, caller_info: Optional[str] = None
):
    """
    Log an exception with traceback

    Parameters
    ----------
        message: str
            The message to log
        exc: Exception, optional
            The exception to log. If None, uses the current exception from sys.exc_info().
        extra: Dict[str, Any], optional
            Additional context to include in the log
        caller_info: str, optional
            Caller information to include in the log. If None, uses the current caller info.

    Returns
    -------
        None

    Raises
    ------
        None
    """

    exc = exc or sys.exc_info()[1]
    exc_info = {
        "error_type": type(exc).__name__ if exc else "Unknown",
        "error_message": str(exc) if exc else "No exception information",
        "error_details": traceback.format_exc(),
    }

    merged_extra = {**(extra or {}), **exc_info}
    error(message, merged_extra, caller_info)


logger = setup_logger()

debug = _log_factory(logging.DEBUG)
info = _log_factory(logging.INFO)
warning = _log_factory(logging.WARNING)
error = _log_factory(logging.ERROR)
critical = _log_factory(logging.CRITICAL)
