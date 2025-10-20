"""
OAMI Logging Configuration Module
=================================

Provides structured JSON logging for the OAMI system.

This module configures the logging subsystem to output structured
JSON logs that can be consumed by external monitoring systems or
analyzed for debugging, metrics, or performance auditing.

The logger automatically writes both to a local file (`./logs/oami.log`)
and to the console for interactive development environments such as Jupyter.

All logging records are filtered to avoid Python logging's reserved attributes
(e.g., 'module', 'msg', 'levelno'), preventing runtime conflicts such as
``KeyError: "Attempt to overwrite 'module' in LogRecord"``.

Examples
--------
>>> from oami.logging_config import setup_json_logging
>>> import logging
>>> setup_json_logging()
>>> logging.info("Data ingestion started.")
>>> logging.warning("Cache not found; fetching data from Polygon API.")
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "oami.log"


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging.

    This formatter serializes log records into JSON strings. It includes
    timestamp, log level, function name, and message fields, as well as
    any additional custom attributes that do not overlap with reserved
    fields of Python's `LogRecord`.

    Notes
    -----
    - The formatter avoids reserved attributes such as `module` or `msg`.
    - It ensures logs remain serializable to JSON format even if complex
      types are included.

    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted.

        Returns
        -------
        str
            A JSON-formatted string representation of the log record.
        """
        # Construct base log structure with essential fields
        log_record: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Dynamically attach non-reserved attributes to the log record
        reserved_keys = set(logging.LogRecord(None, None, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in log_record and key not in reserved_keys:
                try:
                    json.dumps(value)  # test serializability
                    log_record[key] = value
                except TypeError:
                    log_record[key] = str(value)

        # Convert to a compact JSON string
        return json.dumps(log_record, ensure_ascii=False)


def setup_json_logging(log_file: str | None = None) -> None:
    """Initialize structured JSON logging for the OAMI application.

    Parameters
    ----------
    log_file : str, optional
        Path to the log file. Defaults to `'./logs/oami.log'`.

    Raises
    ------
    OSError
        If the log file directory cannot be created.

    Notes
    -----
    This function configures:
        - File logging (persistent storage)
        - Console logging (interactive visibility)
        - JSON-structured output
        - Safe record field handling

    Examples
    --------
    >>> from oami.logging_config import setup_json_logging
    >>> setup_json_logging()
    >>> import logging
    >>> logging.info("OAMI logging initialized successfully.")
    """
    log_path = Path(log_file).resolve() if log_file else DEFAULT_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler with our custom formatter
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(JsonFormatter())

    # Get the root logger and attach handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates in notebooks
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)

    # Log initialization message
    logging.info("Structured JSON logging initialized for OAMI.")


# If this module is executed directly (for testing)
if __name__ == "__main__":
    setup_json_logging()
    logging.info("Logger test event emitted.")
