import time
from datetime import datetime
from .updater import DDNSUpdater
from .advanced_ml_predictor import AdvancedIPPredictor

def main():
    """Main function to run the DDNS updater with ML-powered predictions."""
    updater = DDNSUpdater()
    ip_predictor = AdvancedIPPredictor()
    
    # Base update interval (5 minutes)
    base_interval = 300
    min_interval = 60  # Minimum 1 minute
    max_interval = 3600  # Maximum 1 hour
    
    print("\n🚀 Starting ML-powered DNS-O-Matic updater...")
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            current_time = datetime.now()
            
            # Predict probability of IP change
            change_prob, is_anomaly = ip_predictor.predict_change_probability(current_time)
            
            # Adjust check interval based on prediction
            if is_anomaly:
                print("⚠️  Anomalous pattern detected! Increasing check frequency.")
                update_interval = min_interval
            else:
                # Scale interval inversely with change probability
                # High probability = shorter interval
                update_interval = int(base_interval + (1 - change_prob) * (max_interval - base_interval))
            
            print(f"\n📊 ML Predictions ({current_time.strftime('%Y-%m-%d %H:%M:%S')}):")
            print(f"├── Change Probability: {change_prob:.1%}")
            print(f"├── Anomaly Detected: {'Yes' if is_anomaly else 'No'}")
            print(f"└── Next check in: {update_interval/60:.1f} minutes")
            
            # Perform DNS update
            result = updater.update_dns()
            if result and result.startswith('good'):
                print(f"✅ DNS updated successfully: {result}")
            elif result and result.startswith('nochg'):
                print(f"✓ DNS is up to date: {result}")
            else:
                print(f"❌ DNS update failed: {result}")
            
            # Add a blank line for readability
            print()
            time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n👋 Stopping DNS updater gracefully...")
            if updater.current_ip:
                print(f"Current IP before shutdown: {updater.current_ip}")
            break
        except Exception as e:
            print(f"\n❌ Error in update cycle: {e}")
            print(f"⏰ Using default interval: {base_interval/60:.1f} minutes")
            update_interval = base_interval
        
if __name__ == "__main__":
    main()