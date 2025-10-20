"""
OAMI Configuration Utilities
============================

Handles environment setup, API key loading, and global configuration
for the OAMI framework.

This module centralizes configuration management to ensure consistent
behavior across notebooks, scripts, and deployed services.
"""

import os
from dataclasses import dataclass
import logging
from pathlib import Path
from dotenv import load_dotenv
from oami.logging_config import setup_json_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Settings:
    api_key: str = os.getenv("POLYGON_API_KEY", "YOUR_KEY_HERE")
    cache_root: str = str(PROJECT_ROOT / "data" / "csv")
    default_interval: str = "1D"
    log_file: str = str(PROJECT_ROOT / "logs" / "oami.log")


def initialize_environment() -> str:
    """Initialize OAMI logging and environment configuration.

    This function sets up structured JSON logging and loads the Polygon API key
    from the environment. If the key is not found, a warning is logged and the
    system will operate in offline mode using cached data.

    Notes
    -----
    - Automatically loads `.env` if present.
    - Initializes both console and file logging.
    - Logs clear status messages about API key availability.

    Returns
    -------
    str
        The Polygon API key if found, otherwise `"YOUR_KEY_HERE"`.
    """
    # Load environment variables (supports .env files)
    load_dotenv()

    # Setup structured JSON logging
    setup_json_logging()
    logging.info("OAMI environment initialization started.")

    # Retrieve API key
    api_key = os.getenv("POLYGON_API_KEY", "").strip()

    # Handle missing key gracefully
    if not api_key:
        logging.warning("Polygon API key not found — using offline mode.")
        print("⚠️ No Polygon API key detected. Falling back to cached data.")
        api_key = "YOUR_KEY_HERE"
    else:
        logging.info("Polygon API key loaded successfully.")
        print("✅ Polygon API key detected and loaded.")

    return api_key
