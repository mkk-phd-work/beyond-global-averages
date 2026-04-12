#!/usr/bin/env python
"""Download M5 dataset from Kaggle using API token from .env file."""

import os
import ssl
from pathlib import Path

import requests
import urllib3
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / '.env'
if not env_path.exists():
    raise FileNotFoundError(f".env file not found at {env_path}")

# Load environment variables from .env
load_dotenv(env_path)


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Optional: disable SSL verification for corporate proxy environments.
# Set KAGGLE_SKIP_SSL_VERIFY=true in .env to enable this behavior.
if _env_flag("KAGGLE_SKIP_SSL_VERIFY"):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    _orig_request = requests.sessions.Session.request
    _orig_send = requests.sessions.Session.send

    def _request_no_verify(self, method, url, **kwargs):
        kwargs["verify"] = False
        return _orig_request(self, method, url, **kwargs)

    def _send_no_verify(self, request, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, request, **kwargs)

    requests.sessions.Session.request = _request_no_verify
    requests.sessions.Session.send = _send_no_verify
    print("SSL certificate verification is disabled (KAGGLE_SKIP_SSL_VERIFY=true).")

# Extract Kaggle API token from environment
api_token = os.getenv('KAGGLE_API_TOKEN')
if not api_token:
    raise ValueError("KAGGLE_API_TOKEN not found in .env file or environment")

print(f"Using Kaggle API token: {api_token[:10]}...")

from kaggle.api.kaggle_api_extended import KaggleApi

try:
    # Initialize Kaggle API (will use KAGGLE_API_TOKEN from environment)
    api = KaggleApi()
    api.authenticate()
    print("✓ Authentication successful!")
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download M5 dataset
    print(f"\nDownloading M5 Forecasting Accuracy dataset to {data_dir}...")
    api.competition_download_files(
        'm5-forecasting-accuracy',
        path=str(data_dir)
    )
    
    print("✓ Dataset downloaded successfully!")
    print(f"Files saved to: {data_dir}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for file in sorted(data_dir.glob('m5-forecasting-accuracy*')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")
    for file in sorted(data_dir.glob('*.csv')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")

except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    print(f"\nTroubleshooting:")
    print(f"1. Verify your Kaggle API token is valid")
    print(f"2. Visit: https://www.kaggle.com/account to create a new token if needed")
    print(f"3. Verify the competition is still active: https://www.kaggle.com/competitions/m5-forecasting-accuracy")
    print(f"4. Accept the competition rules at https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules/accept")
    raise
