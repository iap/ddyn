import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DNSOMATIC_USERNAME = os.getenv('DNSOMATIC_USERNAME')
DNSOMATIC_PASSWORD = os.getenv('DNSOMATIC_PASSWORD')
DNSOMATIC_HOSTNAME = os.getenv('DNSOMATIC_HOSTNAME')