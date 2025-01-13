from polygon import RESTClient
from datetime import datetime
import os
from dotenv import load_dotenv


class BaseAnalysis:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("API_KEY")
        self.client = RESTClient(api_key=api_key)

    def validate_dates(self, start_date, end_date):
        """Validate date inputs"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            return start <= end
        except ValueError:
            return False
