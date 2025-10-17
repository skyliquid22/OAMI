import logging, os
from pythonjsonlogger import jsonlogger
def setup_json_logging(log_path: str = "./logs/oami.log") -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fmt = jsonlogger.JsonFormatter(fmt="%(asctime)s %(levelname)s %(name)s %(message)s", json_ensure_ascii=False)
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    root = logging.getLogger(); root.setLevel(logging.INFO); root.handlers = [fh, ch]
    logging.info("JSON logging initialized", extra={"module":"logging_config"})
