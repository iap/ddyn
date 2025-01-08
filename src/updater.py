from requests.auth import HTTPBasicAuth
import requests
from .utils import get_public_ip, predict_ip_change, log_ip_change
from .config import DNSOMATIC_USERNAME, DNSOMATIC_PASSWORD, DNSOMATIC_HOSTNAME

class DDNSUpdater:
    def __init__(self):
        self.username = DNSOMATIC_USERNAME
        self.password = DNSOMATIC_PASSWORD
        self.hostname = DNSOMATIC_HOSTNAME
        self.current_ip = None
        self.url = f"https://updates.dnsomatic.com/nic/update?hostname={self.hostname}"

    def update_dns(self):
        """Update DNS if IP has changed. Returns the response text from DNS-O-Matic."""
        if not predict_ip_change():
            return "nochg (no change predicted)"

        new_ip = get_public_ip()
        if not new_ip:
            return "error (failed to get IP)"

        if new_ip == self.current_ip:
            return f"nochg {new_ip}"

        try:
            response = requests.get(
                self.url,
                params={'myip': new_ip},
                auth=HTTPBasicAuth(self.username, self.password),
                timeout=10
            )
            response.raise_for_status()
            
            # Log the change if update was successful
            if response.text.startswith(('good', 'nochg')):
                log_ip_change(self.current_ip, new_ip)
                self.current_ip = new_ip
            
            return response.text.strip()
            
        except requests.exceptions.Timeout:
            return "error (timeout)"
        except requests.exceptions.ConnectionError:
            return "error (connection failed)"
        except requests.exceptions.HTTPError as e:
            return f"error (HTTP {e.response.status_code})"
        except Exception as e:
            return f"error ({str(e)})"