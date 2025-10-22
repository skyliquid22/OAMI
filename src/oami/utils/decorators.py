import time, random, logging, requests
from polygon import RESTClient
from polygon.exceptions import BadResponse


def retry_request(max_retries: int = 3, base_delay: float = 1.0, jitter: float = 0.3, backoff: float = 2.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    logging.info("Polygon call succeeded", extra={"function": func.__name__, "attempt": attempt})
                    return result
                except (BadResponse, requests.exceptions.RequestException) as e:
                    logging.warning("Polygon API/network error", extra={"function": func.__name__, "attempt": attempt, "error": str(e)})
                except Exception as e:
                    logging.error("Unexpected error", extra={"function": func.__name__, "attempt": attempt, "error": str(e)})
                if attempt < max_retries:
                    sleep_time = delay + random.uniform(0, jitter)
                    logging.info("Retrying Polygon API call", extra={"function": func.__name__, "attempt": attempt, "next_delay": sleep_time})
                    time.sleep(sleep_time); delay *= backoff
            logging.error("Polygon call failed after retries", extra={"function": func.__name__}); return None
        return wrapper
    return decorator
