"""
API Data Loader for SavVio Pipeline
Handles fetching financial and product data from API endpoints.
Supports pagination, rate limiting, and retry logic.
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
from src.utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


class APILoader:
    """Handles data loading from REST API endpoints."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize API client with retry logic.
        
        Args:
            base_url: Base URL for the API
            api_key: API authentication key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Create session with retry strategy
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'SavVio-Pipeline/1.0',
            'Accept': 'application/json'
        })
        
        # Add API key to headers if provided
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
        
        logger.info(f"API client initialized for {self.base_url}")
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to API endpoint.
        
        Args:
            endpoint: API endpoint (e.g., '/financial')
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body data
            
        Returns:
            dict: JSON response data
            
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.info(f"Making {method} request to {url}")
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.timeout
            )
            
            # Raise exception for bad status codes
            response.raise_for_status()
            
            # Parse JSON response
            response_data = response.json()
            
            logger.info(f"Request successful: {response.status_code}")
            return response_data
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            logger.error(f"Response: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response.text}")
            raise
    
    def fetch_with_pagination(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        page_param: str = "page",
        limit_param: str = "limit",
        page_size: int = 100,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from paginated API endpoint.
        
        Args:
            endpoint: API endpoint
            params: Additional query parameters
            page_param: Name of pagination page parameter
            limit_param: Name of pagination limit parameter
            page_size: Number of records per page
            max_pages: Maximum number of pages to fetch (None for all)
            
        Returns:
            list: All records from all pages
        """
        all_records = []
        page = 1
        params = params or {}
        
        logger.info(f"Fetching paginated data from {endpoint}")
        
        while True:
            # Add pagination parameters
            page_params = {
                **params,
                page_param: page,
                limit_param: page_size
            }
            
            try:
                response_data = self._make_request(endpoint, params=page_params)
                
                # Extract records (adjust based on actual API response structure)
                # Common patterns: response_data['data'], response_data['results'], or response_data itself
                if isinstance(response_data, list):
                    records = response_data
                elif 'data' in response_data:
                    records = response_data['data']
                elif 'results' in response_data:
                    records = response_data['results']
                else:
                    records = [response_data]
                
                if not records:
                    logger.info(f"No more records found on page {page}")
                    break
                
                all_records.extend(records)
                logger.info(f"Fetched {len(records)} records from page {page} (total: {len(all_records)})")
                
                # Check if we should continue pagination
                if max_pages and page >= max_pages:
                    logger.info(f"Reached maximum pages limit: {max_pages}")
                    break
                
                # Check if there are more pages (adjust based on API response)
                has_more = False
                if isinstance(response_data, dict):
                    has_more = response_data.get('has_more', False) or \
                              response_data.get('hasMore', False) or \
                              len(records) == page_size
                
                if not has_more:
                    logger.info("No more pages available")
                    break
                
                page += 1
                
                # Rate limiting: small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                raise
        
        logger.info(f"Total records fetched: {len(all_records)}")
        return all_records
    
    def save_to_file(
        self,
        data: List[Dict[str, Any]],
        file_path: str,
        format: str = 'json'
    ) -> str:
        """
        Save fetched data to local file.
        
        Args:
            data: Data to save (list of records)
            file_path: Destination file path
            format: File format ('json' or 'csv')
            
        Returns:
            str: Path to saved file
        """
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving {len(data)} records to {file_path}")
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format.lower() == 'csv':
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Successfully saved {file_size} bytes to {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
            raise
    
    def fetch_and_save(
        self,
        endpoint: str,
        file_path: str,
        format: str = 'json',
        use_pagination: bool = True,
        **pagination_kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from API and save to file, then return as DataFrame.
        
        Args:
            endpoint: API endpoint
            file_path: Destination file path
            format: File format ('json' or 'csv')
            use_pagination: Whether to use pagination
            **pagination_kwargs: Additional arguments for pagination
            
        Returns:
            pd.DataFrame: Fetched data
        """
        try:
            if use_pagination:
                data = self.fetch_with_pagination(endpoint, **pagination_kwargs)
            else:
                response_data = self._make_request(endpoint)
                # Handle single object or list response
                data = response_data if isinstance(response_data, list) else [response_data]
            
            # Save to file
            self.save_to_file(data, file_path, format)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            logger.info(f"Converted to DataFrame: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch and save data: {e}")
            raise


# Convenience functions for SavVio data pipeline

def load_financial_data(
    api_base_url: str,
    endpoint: str = "/financial",
    destination_path: str = "data/raw/financial_data.csv",
    api_key: Optional[str] = None,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Load financial data from API endpoint.
    
    Args:
        api_base_url: Base URL for the API
        endpoint: API endpoint for financial data
        destination_path: Local path to save data
        api_key: API authentication key
        timeout: Request timeout in seconds
        
    Returns:
        pd.DataFrame: Financial data
    """
    logger.info("=" * 60)
    logger.info("Loading Financial Data from API")
    logger.info("=" * 60)
    
    loader = APILoader(base_url=api_base_url, api_key=api_key, timeout=timeout)
    
    # Financial data is typically CSV format
    df = loader.fetch_and_save(
        endpoint=endpoint,
        file_path=destination_path,
        format='csv',
        use_pagination=True,
        page_size=100
    )
    
    logger.info(f"Financial data loaded successfully: {df.shape}")
    return df


def load_product_data(
    api_base_url: str,
    endpoint: str = "/products",
    destination_path: str = "data/raw/product_data.json",
    api_key: Optional[str] = None,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Load product data from API endpoint.
    
    Args:
        api_base_url: Base URL for the API
        endpoint: API endpoint for product data
        destination_path: Local path to save data
        api_key: API authentication key
        timeout: Request timeout in seconds
        
    Returns:
        pd.DataFrame: Product data
    """
    logger.info("=" * 60)
    logger.info("Loading Product Data from API")
    logger.info("=" * 60)
    
    loader = APILoader(base_url=api_base_url, api_key=api_key, timeout=timeout)
    
    # Product data is JSON format
    df = loader.fetch_and_save(
        endpoint=endpoint,
        file_path=destination_path,
        format='json',
        use_pagination=True,
        page_size=100
    )
    
    logger.info(f"Product data loaded successfully: {df.shape}")
    return df


def load_review_data(
    api_base_url: str,
    endpoint: str = "/reviews",
    destination_path: str = "data/raw/review_data.json",
    api_key: Optional[str] = None,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Load product review data from API endpoint.
    
    Args:
        api_base_url: Base URL for the API
        endpoint: API endpoint for review data
        destination_path: Local path to save data
        api_key: API authentication key
        timeout: Request timeout in seconds
        
    Returns:
        pd.DataFrame: Review data
    """
    logger.info("=" * 60)
    logger.info("Loading Product Review Data from API")
    logger.info("=" * 60)
    
    loader = APILoader(base_url=api_base_url, api_key=api_key, timeout=timeout)
    
    # Review data is JSON format
    df = loader.fetch_and_save(
        endpoint=endpoint,
        file_path=destination_path,
        format='json',
        use_pagination=True,
        page_size=100
    )
    
    logger.info(f"Review data loaded successfully: {df.shape}")
    return df


# Example usage and testing
if __name__ == "__main__":
    # This will be replaced by config.py in actual usage
    from dotenv import load_dotenv
    load_dotenv()
    
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.savvio.com/v1")
    API_KEY = os.getenv("API_KEY")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    
    try:
        # Example: Load financial data
        financial_df = load_financial_data(
            api_base_url=API_BASE_URL,
            api_key=API_KEY,
            timeout=API_TIMEOUT
        )
        print("\nFinancial Data Preview:")
        print(financial_df.head())
        
        # Example: Load product data
        product_df = load_product_data(
            api_base_url=API_BASE_URL,
            api_key=API_KEY,
            timeout=API_TIMEOUT
        )
        print("\nProduct Data Preview:")
        print(product_df.head())
        
        # Example: Load review data
        review_df = load_review_data(
            api_base_url=API_BASE_URL,
            api_key=API_KEY,
            timeout=API_TIMEOUT
        )
        print("\nReview Data Preview:")
        print(review_df.head())
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise