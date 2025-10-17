import os
from dataclasses import dataclass
@dataclass
class Settings:
    api_key: str = os.getenv("POLYGON_API_KEY", "YOUR_KEY_HERE")
    cache_root: str = "./data/csv"
    timeframe: str = "day"
    log_file: str = "./logs/oami.log"
