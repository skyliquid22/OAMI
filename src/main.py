from oami.logging_config import setup_json_logging
from oami.__version__ import __version__
import logging

if __name__ == "__main__":
    setup_json_logging()
    logging.info("Starting OAMI pipeline", extra={"version": __version__})
    # Prefer offline sample data if available
    pass
