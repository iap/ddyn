import requests
import pandas as pd
from datetime import datetime
import os
import random
import time
from .advanced_ml_predictor import AdvancedIPPredictor

# Initialize ML predictor
ip_predictor = AdvancedIPPredictor()

# Privacy-focused DoH services
DOH_SERVICES = [
    {
        'url': "https://cloudflare-dns.com/dns-query",
        'name': "cloudflare",
        'headers': {
            'accept': 'application/dns-json'
        }
    },
    {
        'url': "https://dns.google/resolve",  # Changed to resolve endpoint
        'name': "google",
        'headers': {
            'accept': 'application/dns-json'
        }
    }
]

def get_public_ip_doh():
    """Get public IP using privacy-focused DNS over HTTPS services."""
    for service in DOH_SERVICES:
        try:
            params = {
                'name': 'myip.opendns.com',
                'type': 'A'
            }
            response = requests.get(
                service['url'],
                headers=service['headers'],
                params=params,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            if 'Answer' in data:
                for answer in data['Answer']:
                    if is_valid_ip(answer.get('data')):
                        return answer['data']
        except Exception as e:
            continue  # Silently try next service
    
    return None

def predict_ip_change():
    """Predict if an IP change is likely based on ML model."""
    try:
        change_prob, is_anomaly = ip_predictor.predict_change_probability()
        return change_prob > 0.5 or is_anomaly
    except Exception as e:
        print(f"Error predicting IP change: {e}")
        return False

def log_ip_change(old_ip, new_ip, retention_days=30):
    """Log IP changes to parquet file and maintain data retention policy."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Prepare new data
        new_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'old_ip': [old_ip],
            'new_ip': [new_ip],
            'ip_changed': [old_ip != new_ip]
        })
        
        # Load existing data or create new file
        data_file = 'data/ip_changes.parquet'
        if os.path.exists(data_file):
            df = pd.read_parquet(data_file)
            # Remove old data
            cutoff_date = datetime.now() - pd.Timedelta(days=retention_days)
            df = df[df['timestamp'] >= cutoff_date]
            # Append new data
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        
        # Save updated data
        df.to_parquet(data_file, index=False)
        
        # Retrain model with new data if needed
        if len(df) > 24:  # Need at least 24 hours of data
            ip_predictor.train(data_file)
            
    except Exception as e:
        print(f"Error logging IP change: {e}")

def is_valid_ip(ip):
    """Validate IPv4 address format."""
    if not ip:
        return False
    parts = ip.split('.')
    return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)

# Privacy-focused IP lookup services
IP_SERVICES = [
    {
        'url': 'https://icanhazip.com',  # No logging, simple text response
        'headers': {'User-Agent': 'curl/7.64.1'}
    },
    {
        'url': 'https://api.ipify.org',  # Privacy-focused, simple text response
        'headers': {'User-Agent': 'curl/7.64.1'}
    },
    {
        'url': 'https://ifconfig.me/ip',  # Privacy-focused, simple text response
        'headers': {'User-Agent': 'curl/7.64.1'}
    }
]

def get_public_ip_direct():
    """Get public IP using privacy-focused IP lookup services."""
    for service in IP_SERVICES:
        try:
            response = requests.get(
                service['url'],
                headers=service['headers'],
                timeout=5
            )
            response.raise_for_status()
            ip = response.text.strip()
            if is_valid_ip(ip):
                return ip
        except Exception as e:
            print(f"Error with IP service {service['url']}: {e}")
            continue
    
    return None

def get_public_ip():
    """Get the current public IP address using privacy-focused methods."""
    # Try DNS over HTTPS first (more privacy-focused)
    ip = get_public_ip_doh()
    if ip:
        return ip
        
    # Fall back to direct IP services if DoH fails
    ip = get_public_ip_direct()
    if ip:
        return ip
    
    raise Exception("Failed to get public IP from all services")