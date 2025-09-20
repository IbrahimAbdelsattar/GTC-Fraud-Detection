import logging
import logging.config
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from src.config.parameters import LOGGING_CONFIG, LOGS_DIR

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Model monitoring and performance tracking"""
    
    def __init__(self, log_file: str = "model_predictions.log"):
        self.log_file = LOGS_DIR / log_file
        self.prediction_logs = []
        
    def log_prediction(self, input_data: Dict, prediction: int, probability: float, 
                      processing_time: float, model_version: str = "1.0"):
        """Log individual prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'input_features': {k: v for k, v in input_data.items() if k != 'Class'},
            'prediction': prediction,
            'fraud_probability': probability,
            'processing_time_ms': processing_time * 1000,
            'prediction_class': 'Fraud' if prediction == 1 else 'Legitimate'
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        self.prediction_logs.append(log_entry)
        
        logger.info(f"Prediction logged: {log_entry['prediction_class']} "
                   f"(prob: {probability:.4f}, time: {processing_time*1000:.2f}ms)")
    
    def load_prediction_logs(self, days_back: int = 30) -> pd.DataFrame:
        """Load prediction logs for analysis"""
        try:
            logs = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
            
            df_logs = pd.DataFrame(logs)
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])

            cutoff_date = datetime.now() - pd.Timedelta(days=days_back)
            df_logs = df_logs[df_logs['timestamp'] >= cutoff_date]
            
            logger.info(f"Loaded {len(df_logs)} prediction logs from last {days_back} days")
            return df_logs
            
        except FileNotFoundError:
            logger.warning("No prediction log file found")
            return pd.DataFrame()
    
    def generate_monitoring_report(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate monitoring report"""
        df_logs = self.load_prediction_logs(days_back)
        
        if df_logs.empty:
            return {"error": "No data available for monitoring report"}
        
        report = {
            'period': f"Last {days_back} days",
            'total_predictions': len(df_logs),
            'fraud_predictions': df_logs['prediction'].sum(),
            'fraud_rate': df_logs['prediction'].mean(),
            'avg_processing_time': df_logs['processing_time_ms'].mean(),
            'max_processing_time': df_logs['processing_time_ms'].max(),
            'min_processing_time': df_logs['processing_time_ms'].min(),
            'avg_fraud_probability': df_logs['fraud_probability'].mean(),
            'high_confidence_predictions': len(df_logs[df_logs['fraud_probability'] > 0.8]),
            'low_confidence_predictions': len(df_logs[df_logs['fraud_probability'] < 0.2])
        }

        daily_stats = df_logs.groupby(df_logs['timestamp'].dt.date).agg({
            'prediction': ['count', 'sum', 'mean'],
            'processing_time_ms': 'mean',
            'fraud_probability': 'mean'
        }).round(4)
        
        report['daily_statistics'] = daily_stats.to_dict()
        
        logger.info(f"Generated monitoring report for {days_back} days")
        return report
    
    def plot_monitoring_dashboard(self, days_back: int = 7):
        """Create monitoring dashboard plots"""
        df_logs = self.load_prediction_logs(days_back)
        
        if df_logs.empty:
            logger.warning("No data for monitoring dashboard")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Monitoring Dashboard - Last {days_back} Days', fontsize=16)

        daily_counts = df_logs.groupby(df_logs['timestamp'].dt.date)['prediction'].agg(['count', 'sum'])
        axes[0, 0].plot(daily_counts.index, daily_counts['count'], marker='o', label='Total')
        axes[0, 0].plot(daily_counts.index, daily_counts['sum'], marker='s', label='Fraud')
        axes[0, 0].set_title('Daily Prediction Counts')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].hist(df_logs['fraud_probability'], bins=20, alpha=0.7, color='orange')
        axes[0, 1].set_title('Fraud Probability Distribution')
        axes[0, 1].set_xlabel('Fraud Probability')
        axes[0, 1].set_ylabel('Frequency')

        axes[1, 0].scatter(df_logs['timestamp'], df_logs['processing_time_ms'], alpha=0.6)
        axes[1, 0].set_title('Processing Time Trend')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Processing Time (ms)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        df_logs['hour'] = df_logs['timestamp'].dt.hour
        hourly_stats = df_logs.groupby('hour')['prediction'].agg(['count', 'mean'])
        axes[1, 1].bar(hourly_stats.index, hourly_stats['count'], alpha=0.7, label='Count')
        ax2 = axes[1, 1].twinx()
        ax2.plot(hourly_stats.index, hourly_stats['mean'], color='red', marker='o', label='Fraud Rate')
        axes[1, 1].set_title('Hourly Prediction Pattern')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Prediction Count')
        ax2.set_ylabel('Fraud Rate')
        
        plt.tight_layout()

        dashboard_path = LOGS_DIR / f"monitoring_dashboard_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Monitoring dashboard saved to {dashboard_path}")
        return dashboard_path


class DataDriftMonitor:
    """Monitor data drift in production"""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_stats = self._calculate_stats(reference_data)
        
    def _calculate_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate reference statistics"""
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q50': df[col].quantile(0.50),
                'q75': df[col].quantile(0.75)
            }
        return stats
    
    def detect_drift(self, current_data: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """Detect data drift compared to reference data"""
        current_stats = self._calculate_stats(current_data)
        drift_results = {}
        
        for col in self.reference_stats.keys():
            if col in current_stats:
                ref_mean = self.reference_stats[col]['mean']
                cur_mean = current_stats[col]['mean']
                ref_std = self.reference_stats[col]['std']

                drift_score = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
                
                drift_results[col] = {
                    'drift_score': drift_score,
                    'is_drifted': drift_score > threshold,
                    'reference_mean': ref_mean,
                    'current_mean': cur_mean,
                    'relative_change': (cur_mean - ref_mean) / (ref_mean + 1e-8)
                }

        drifted_features = [col for col, result in drift_results.items() if result['is_drifted']]
        avg_drift_score = np.mean([result['drift_score'] for result in drift_results.values()])
        
        summary = {
            'total_features_monitored': len(drift_results),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'average_drift_score': avg_drift_score,
            'overall_drift_detected': len(drifted_features) > 0,
            'feature_details': drift_results
        }
        
        logger.info(f"Data drift analysis: {len(drifted_features)} drifted features out of {len(drift_results)}")
        return summary


def setup_logging():
    """Setup logging configuration"""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")
    return logger


def log_model_performance(metrics: Dict[str, float], model_name: str):
    """Log model performance metrics"""
    logger.info(f"Model Performance - {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


def create_alert(alert_type: str, message: str, severity: str = "INFO"):
    """Create monitoring alert"""
    alert = {
        'timestamp': datetime.now().isoformat(),
        'alert_type': alert_type,
        'severity': severity,
        'message': message
    }

    if severity == "CRITICAL":
        logger.critical(f"ALERT [{alert_type}]: {message}")
    elif severity == "WARNING":
        logger.warning(f"ALERT [{alert_type}]: {message}")
    else:
        logger.info(f"ALERT [{alert_type}]: {message}")

    alerts_file = LOGS_DIR / "alerts.log"
    with open(alerts_file, 'a') as f:
        f.write(json.dumps(alert) + '\n')
    
    return alert