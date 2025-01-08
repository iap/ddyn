import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(days=30, change_probability=0.1):
    """Generate sample IP change data for training."""
    # Create directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate timestamps for the past N days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Generate hourly timestamps
    timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Create base dataframe
    df = pd.DataFrame({
        'timestamp': timestamps
    })
    
    # Generate IP changes with patterns
    changes = []
    last_change = False
    
    for hour in df['timestamp'].dt.hour:
        # Increase probability during certain hours (e.g., maintenance windows)
        if 2 <= hour <= 4:  # Maintenance window
            prob = change_probability * 3
        elif 22 <= hour or hour <= 5:  # Night time
            prob = change_probability * 2
        else:
            prob = change_probability
            
        # Reduce probability if there was a recent change
        if last_change:
            prob *= 0.2
            
        # Generate change
        change = np.random.random() < prob
        changes.append(change)
        last_change = change
    
    df['ip_changed'] = changes
    
    # Add some weekly patterns
    weekly_pattern = np.sin(2 * np.pi * df.index / (7 * 24)) + 1
    weekly_changes = np.random.random(len(df)) < (change_probability * weekly_pattern)
    df['ip_changed'] = df['ip_changed'] | weekly_changes
    
    # Generate some IP addresses
    ips = []
    current_ip = '192.168.1.1'
    
    for change in df['ip_changed']:
        if change:
            # Generate a new random IP
            new_ip = '.'.join(str(np.random.randint(1, 255)) for _ in range(4))
            current_ip = new_ip
        ips.append(current_ip)
    
    df['ip_address'] = ips
    
    # Save to parquet file
    df.to_parquet('data/ip_changes.parquet', index=False)
    
    print(f"Generated {len(df)} records spanning {days} days")
    print(f"Total IP changes: {df['ip_changed'].sum()}")
    print(f"Change rate: {df['ip_changed'].mean():.1%}")
    
    return df

if __name__ == '__main__':
    # Generate 30 days of sample data
    generate_sample_data(days=30, change_probability=0.1)
