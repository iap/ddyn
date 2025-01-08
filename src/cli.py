import click
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
from .advanced_ml_predictor import AdvancedIPPredictor

@click.group()
def cli():
    """ML Model Management CLI for DNS Updater"""
    pass

@cli.command()
@click.option('--force', is_flag=True, help='Force retrain even if model exists')
def train(force):
    """Train the ML models on collected data"""
    predictor = AdvancedIPPredictor()
    
    if not force and Path(predictor.model_path).exists():
        click.echo("Model already exists. Use --force to retrain.")
        return
        
    click.echo("Training ML models...")
    success = predictor.train()
    
    if success:
        click.echo("✅ Training completed successfully!")
    else:
        click.echo("❌ Training failed. Check logs for details.")

@cli.command()
@click.option('--days', default=7, help='Number of days to evaluate')
def evaluate(days):
    """Evaluate model performance on recent data"""
    predictor = AdvancedIPPredictor()
    
    try:
        df = pd.read_parquet('data/ip_changes.parquet')
        end_date = df['timestamp'].max()
        start_date = end_date - timedelta(days=days)
        
        df_eval = df[df['timestamp'] >= start_date]
        
        results = {
            'total_predictions': len(df_eval),
            'correct_predictions': 0,
            'anomalies_detected': 0,
            'average_probability': 0.0
        }
        
        for _, row in df_eval.iterrows():
            prob, is_anomaly = predictor.predict_change_probability(row['timestamp'])
            if (prob > 0.5) == row['ip_changed']:
                results['correct_predictions'] += 1
            if is_anomaly:
                results['anomalies_detected'] += 1
            results['average_probability'] += prob
            
        results['average_probability'] /= len(df_eval)
        results['accuracy'] = results['correct_predictions'] / len(df_eval)
        
        click.echo("\n📊 Model Evaluation Results:")
        click.echo(f"Period: Last {days} days ({start_date.date()} to {end_date.date()})")
        click.echo(f"Total Predictions: {results['total_predictions']}")
        click.echo(f"Accuracy: {results['accuracy']:.1%}")
        click.echo(f"Anomalies Detected: {results['anomalies_detected']}")
        click.echo(f"Average Change Probability: {results['average_probability']:.1%}")
        
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {str(e)}")

@cli.command()
@click.argument('timestamp', required=False)
def predict(timestamp):
    """Get prediction for a specific timestamp (default: now)"""
    predictor = AdvancedIPPredictor()
    
    if timestamp:
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            click.echo("❌ Invalid timestamp format. Use ISO format (YYYY-MM-DD HH:MM:SS)")
            return
    else:
        timestamp = datetime.now()
    
    prob, is_anomaly = predictor.predict_change_probability(timestamp)
    
    click.echo("\n🔮 Prediction Results:")
    click.echo(f"Timestamp: {timestamp}")
    click.echo(f"Change Probability: {prob:.1%}")
    click.echo(f"Anomaly Detected: {'Yes' if is_anomaly else 'No'}")

@cli.command()
def status():
    """Show model status and statistics"""
    predictor = AdvancedIPPredictor()
    
    try:
        model_path = Path(predictor.model_path)
        stats = {
            'model_exists': model_path.exists(),
            'model_size': model_path.stat().st_size if model_path.exists() else 0,
            'last_modified': datetime.fromtimestamp(model_path.stat().st_mtime) if model_path.exists() else None,
        }
        
        # Get training data stats
        df = pd.read_parquet('data/ip_changes.parquet')
        data_stats = {
            'total_records': len(df),
            'date_range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
            'total_changes': df['ip_changed'].sum(),
            'change_rate': df['ip_changed'].mean()
        }
        
        click.echo("\n📈 Model Status:")
        click.echo(f"Model Exists: {'Yes' if stats['model_exists'] else 'No'}")
        if stats['model_exists']:
            click.echo(f"Last Modified: {stats['last_modified']}")
            click.echo(f"Model Size: {stats['model_size'] / 1024:.1f} KB")
        
        click.echo("\n📊 Training Data Statistics:")
        click.echo(f"Total Records: {data_stats['total_records']}")
        click.echo(f"Date Range: {data_stats['date_range']}")
        click.echo(f"Total IP Changes: {data_stats['total_changes']}")
        click.echo(f"Change Rate: {data_stats['change_rate']:.1%}")
        
    except Exception as e:
        click.echo(f"❌ Error getting status: {str(e)}")

@cli.command()
@click.option('--output', type=click.Path(), help='Output file path')
def export_metrics(output):
    """Export model metrics and predictions to JSON"""
    predictor = AdvancedIPPredictor()
    
    try:
        df = pd.read_parquet('data/ip_changes.parquet')
        last_week = df['timestamp'].max() - timedelta(days=7)
        df_recent = df[df['timestamp'] >= last_week]
        
        metrics = {
            'predictions': [],
            'summary': {
                'total_records': len(df_recent),
                'period_start': df_recent['timestamp'].min().isoformat(),
                'period_end': df_recent['timestamp'].max().isoformat()
            }
        }
        
        for _, row in df_recent.iterrows():
            prob, is_anomaly = predictor.predict_change_probability(row['timestamp'])
            metrics['predictions'].append({
                'timestamp': row['timestamp'].isoformat(),
                'actual_change': bool(row['ip_changed']),
                'predicted_probability': float(prob),
                'is_anomaly': bool(is_anomaly)
            })
        
        if output:
            with open(output, 'w') as f:
                json.dump(metrics, f, indent=2)
            click.echo(f"✅ Metrics exported to {output}")
        else:
            click.echo(json.dumps(metrics, indent=2))
            
    except Exception as e:
        click.echo(f"❌ Error exporting metrics: {str(e)}")

if __name__ == '__main__':
    cli()
