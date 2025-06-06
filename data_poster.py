import logging
import time
import requests
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class RobustDataPoster:
    """Enhanced data posting with retry mechanisms - No offline storage"""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.timeout = 15
        
    def post_with_exponential_backoff(self, url: str, data: Dict = None, files: Dict = None, 
                                    json_data: Dict = None) -> Tuple[bool, Optional[Dict]]:
        """Post data with exponential backoff retry strategy"""
        for attempt in range(self.max_retries):
            try:
                delay = self.base_delay * (2 ** attempt)
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay}s delay")
                    time.sleep(delay)
                if files:
                    response = self.session.post(url, data=data, files=files)
                elif json_data:
                    response = self.session.post(url, json=json_data, 
                                               headers={'Content-Type': 'application/json'})
                else:
                    response = self.session.post(url, data=data)
                response.raise_for_status()
                try:
                    return True, response.json()
                except:
                    return True, {"status": "success", "message": "Posted successfully"}
            except requests.exceptions.Timeout:
                logger.error(f"Timeout on attempt {attempt + 1}")
                continue
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error on attempt {attempt + 1}")
                continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [400, 401, 403, 404]:
                    logger.error(f"Client error {e.response.status_code}: {e.response.text}")
                    return False, {"error": f"Client error: {e.response.status_code}"}
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                continue
        return False, {"error": "Max retries exceeded"}
    
    def get_with_exponential_backoff(self, url: str, params: Dict = None) -> Tuple[bool, Optional[Dict]]:
        """GET request with exponential backoff retry strategy"""
        for attempt in range(self.max_retries):
            try:
                delay = self.base_delay * (2 ** attempt)
                if attempt > 0:
                    logger.info(f"GET retry attempt {attempt + 1}/{self.max_retries} after {delay}s delay")
                    time.sleep(delay)
                response = self.session.get(url, params=params)
                response.raise_for_status()
                try:
                    return True, response.json()
                except:
                    return True, {"status": "success", "message": "Request successful"}
            except requests.exceptions.Timeout:
                logger.error(f"GET timeout on attempt {attempt + 1}")
                continue
            except requests.exceptions.ConnectionError:
                logger.error(f"GET connection error on attempt {attempt + 1}")
                continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [400, 401, 403, 404]:
                    logger.error(f"GET client error {e.response.status_code}: {e.response.text}")
                    return False, {"error": f"Client error: {e.response.status_code}"}
                logger.error(f"GET HTTP error on attempt {attempt + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"GET unexpected error on attempt {attempt + 1}: {e}")
                continue
        return False, {"error": "Max retries exceeded"}