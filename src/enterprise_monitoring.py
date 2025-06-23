#!/usr/bin/env python3
"""
Enterprise Monitoring and Analytics System
===========================================

Advanced monitoring capabilities for production deepfake detection systems:
- Real-time performance tracking
- Model accuracy monitoring
- Automated alerting
- Resource usage monitoring
- Fraud detection patterns
- Business intelligence metrics

For enterprise deployments requiring comprehensive monitoring.
"""

import psutil
import sqlite3
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import requests
import logging
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enterprise_monitoring.log'),
        logging.StreamHandler()
    ]
)

class EnterpriseMonitor:
    """
    Enterprise-grade monitoring system for deepfake detection
    """
    
    def __init__(self, db_path="analytics/analytics.db", config_path="config/monitoring.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.config = self.load_config()
        
        # Ensure directories exist
        Path("logs").mkdir(exist_ok=True)
        Path("config").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.init_monitoring_tables()
    
    def load_config(self) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            "alerts": {
                "enabled": True,
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": "",
                    "recipients": []
                },
                "thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 80,
                    "error_rate": 5,
                    "response_time": 2000,
                    "accuracy_drop": 10
                }
            },
            "monitoring": {
                "interval_seconds": 60,
                "retention_days": 90,
                "api_endpoint": "http://localhost:8001"
            },
            "business_intelligence": {
                "fraud_detection": True,
                "usage_analytics": True,
                "performance_trends": True
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # Create default config
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            self.logger.warning(f"Failed to load config, using defaults: {e}")
            return default_config
    
    def init_monitoring_tables(self):
        """Initialize additional monitoring tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # System health metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage REAL,
                    network_io_sent INTEGER,
                    network_io_recv INTEGER,
                    active_connections INTEGER,
                    response_time_ms REAL
                )
            ''')
            
            # Model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accuracy_rate REAL,
                    false_positive_rate REAL,
                    false_negative_rate REAL,
                    throughput_per_minute REAL,
                    avg_confidence REAL,
                    model_version TEXT
                )
            ''')
            
            # Alert history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL,
                    threshold_value REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME
                )
            ''')
            
            # Business metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS business_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_revenue REAL,
                    api_calls_billable INTEGER,
                    unique_users INTEGER,
                    enterprise_clients INTEGER,
                    fraud_attempts_detected INTEGER,
                    geographic_distribution TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Monitoring tables initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring tables: {e}")
    
    def collect_system_metrics(self) -> Dict:
        """Collect comprehensive system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Process information
            active_connections = len(psutil.net_connections())
            
            # API Response time (if service is running)
            response_time = self.check_api_response_time()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'network_io_sent': network_io.bytes_sent,
                'network_io_recv': network_io.bytes_recv,
                'active_connections': active_connections,
                'response_time_ms': response_time
            }
            
            # Store in database
            self.store_system_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def check_api_response_time(self) -> float:
        """Check API response time"""
        try:
            api_url = f"{self.config['monitoring']['api_endpoint']}/health"
            start_time = time.time()
            response = requests.get(api_url, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return response_time
            else:
                self.logger.warning(f"API health check failed: {response.status_code}")
                return -1
                
        except requests.RequestException as e:
            self.logger.warning(f"API unavailable: {e}")
            return -1
    
    def store_system_metrics(self, metrics: Dict):
        """Store system metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_health (
                    cpu_percent, memory_percent, disk_usage, 
                    network_io_sent, network_io_recv, active_connections, response_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.get('cpu_percent'),
                metrics.get('memory_percent'),
                metrics.get('disk_usage_percent'),
                metrics.get('network_io_sent'),
                metrics.get('network_io_recv'),
                metrics.get('active_connections'),
                metrics.get('response_time_ms')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store system metrics: {e}")
    
    def analyze_model_performance(self) -> Dict:
        """Analyze current model performance trends"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get last 24 hours of predictions
            cutoff = datetime.now() - timedelta(hours=24)
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN success = 1 THEN 1 END) as successful,
                    AVG(confidence) as avg_confidence,
                    AVG(inference_time_ms) as avg_inference_time,
                    COUNT(CASE WHEN prediction = 'real' AND success = 1 THEN 1 END) as real_predictions,
                    COUNT(CASE WHEN prediction = 'fake' AND success = 1 THEN 1 END) as fake_predictions
                FROM predictions 
                WHERE timestamp >= ?
            ''', (cutoff,))
            
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                total, successful, avg_confidence, avg_inference_time, real_pred, fake_pred = result
                
                performance = {
                    'total_predictions': total,
                    'success_rate': (successful / total) * 100,
                    'avg_confidence': avg_confidence or 0,
                    'avg_inference_time': avg_inference_time or 0,
                    'real_fake_ratio': real_pred / max(fake_pred, 1),
                    'throughput_per_minute': total / (24 * 60),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store performance metrics
                self.store_model_performance(performance)
                
                conn.close()
                return performance
            else:
                conn.close()
                return {'message': 'No recent predictions to analyze'}
                
        except Exception as e:
            self.logger.error(f"Failed to analyze model performance: {e}")
            return {'error': str(e)}
    
    def store_model_performance(self, performance: Dict):
        """Store model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance (
                    accuracy_rate, throughput_per_minute, avg_confidence, model_version
                ) VALUES (?, ?, ?, ?)
            ''', (
                performance.get('success_rate'),
                performance.get('throughput_per_minute'),
                performance.get('avg_confidence'),
                'retrained_v1.0'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store model performance: {e}")
    
    def check_alert_conditions(self, system_metrics: Dict, model_performance: Dict):
        """Check for alert conditions and trigger notifications"""
        alerts = []
        thresholds = self.config['alerts']['thresholds']
        
        # System alerts
        if system_metrics.get('cpu_percent', 0) > thresholds['cpu_usage']:
            alerts.append({
                'type': 'system',
                'severity': 'warning',
                'message': f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                'metric_value': system_metrics['cpu_percent'],
                'threshold_value': thresholds['cpu_usage']
            })
        
        if system_metrics.get('memory_percent', 0) > thresholds['memory_usage']:
            alerts.append({
                'type': 'system',
                'severity': 'warning',
                'message': f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                'metric_value': system_metrics['memory_percent'],
                'threshold_value': thresholds['memory_usage']
            })
        
        if system_metrics.get('response_time_ms', 0) > thresholds['response_time']:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"Slow API response: {system_metrics['response_time_ms']:.1f}ms",
                'metric_value': system_metrics['response_time_ms'],
                'threshold_value': thresholds['response_time']
            })
        
        # Model performance alerts
        if model_performance.get('success_rate', 100) < (100 - thresholds['error_rate']):
            alerts.append({
                'type': 'model',
                'severity': 'critical',
                'message': f"High error rate: {100 - model_performance['success_rate']:.1f}%",
                'metric_value': 100 - model_performance['success_rate'],
                'threshold_value': thresholds['error_rate']
            })
        
        # Process alerts
        for alert in alerts:
            self.process_alert(alert)
        
        return alerts
    
    def process_alert(self, alert: Dict):
        """Process and store alert, send notifications if configured"""
        try:
            # Store alert in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (
                    alert_type, severity, message, metric_value, threshold_value
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                alert['type'],
                alert['severity'],
                alert['message'],
                alert.get('metric_value'),
                alert.get('threshold_value')
            ))
            
            conn.commit()
            conn.close()
            
            # Log alert
            self.logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")
            
            # Send email notification if configured
            if self.config['alerts']['enabled'] and self.config['alerts']['email']['recipients']:
                self.send_email_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to process alert: {e}")
    
    def send_email_alert(self, alert: Dict):
        """Send email alert notification"""
        try:
            email_config = self.config['alerts']['email']
            
            if not email_config['sender_email'] or not email_config['sender_password']:
                self.logger.warning("Email credentials not configured")
                return
            
            msg = MimeMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"🚨 Deepfake Detection Alert - {alert['severity'].upper()}"
            
            body = f"""
            Alert Details:
            
            Type: {alert['type']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            Timestamp: {datetime.now().isoformat()}
            
            Current Value: {alert.get('metric_value', 'N/A')}
            Threshold: {alert.get('threshold_value', 'N/A')}
            
            Please check the enterprise dashboard for more details.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            text = msg.as_string()
            server.sendmail(email_config['sender_email'], email_config['recipients'], text)
            server.quit()
            
            self.logger.info(f"Alert email sent to {len(email_config['recipients'])} recipients")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def generate_business_intelligence_report(self) -> Dict:
        """Generate comprehensive business intelligence report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Last 30 days
            cutoff = datetime.now() - timedelta(days=30)
            
            # Usage analytics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_api_calls,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN prediction = 'fake' THEN 1 END) as fake_detected,
                    COUNT(CASE WHEN prediction = 'real' THEN 1 END) as real_validated
                FROM predictions 
                WHERE timestamp >= ? AND success = 1
            ''', (cutoff,))
            
            usage_stats = cursor.fetchone()
            
            # Fraud detection patterns
            cursor.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(CASE WHEN prediction = 'fake' AND confidence > 90 THEN 1 END) as high_confidence_fakes,
                    COUNT(*) as total_daily
                FROM predictions 
                WHERE timestamp >= ? AND success = 1
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
            ''', (cutoff,))
            
            fraud_patterns = cursor.fetchall()
            
            # Performance trends
            cursor.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(inference_time_ms) as avg_response_time,
                    COUNT(*) as volume
                FROM predictions 
                WHERE timestamp >= ? AND success = 1
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''', (cutoff,))
            
            performance_trends = cursor.fetchall()
            
            conn.close()
            
            # Calculate business metrics
            if usage_stats and usage_stats[0]:
                total_calls, active_days, avg_confidence, fake_detected, real_validated = usage_stats
                
                estimated_monthly_revenue = total_calls * 0.05  # $0.05 per API call
                fraud_prevention_value = fake_detected * 1000  # $1000 per prevented fraud
                
                report = {
                    'summary': {
                        'total_api_calls': total_calls,
                        'active_days': active_days,
                        'avg_confidence': round(avg_confidence or 0, 2),
                        'fake_images_detected': fake_detected,
                        'real_images_validated': real_validated,
                        'estimated_monthly_revenue': round(estimated_monthly_revenue, 2),
                        'fraud_prevention_value': fraud_prevention_value
                    },
                    'fraud_patterns': [
                        {
                            'date': row[0],
                            'high_confidence_fakes': row[1],
                            'total_daily': row[2],
                            'fraud_rate': round((row[1] / max(row[2], 1)) * 100, 2)
                        } for row in fraud_patterns
                    ],
                    'performance_trends': [
                        {
                            'date': row[0],
                            'avg_response_time': round(row[1] or 0, 2),
                            'volume': row[2]
                        } for row in performance_trends
                    ],
                    'generated_at': datetime.now().isoformat()
                }
                
                return report
            else:
                return {'message': 'Insufficient data for business intelligence report'}
                
        except Exception as e:
            self.logger.error(f"Failed to generate BI report: {e}")
            return {'error': str(e)}
    
    def run_monitoring_cycle(self):
        """Execute one complete monitoring cycle"""
        try:
            self.logger.info("Starting monitoring cycle...")
            
            # Collect system metrics
            system_metrics = self.collect_system_metrics()
            
            # Analyze model performance
            model_performance = self.analyze_model_performance()
            
            # Check for alerts
            alerts = self.check_alert_conditions(system_metrics, model_performance)
            
            # Log status
            self.logger.info(f"Monitoring cycle complete - {len(alerts)} alerts generated")
            
            return {
                'system_metrics': system_metrics,
                'model_performance': model_performance,
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring cycle failed: {e}")
            return {'error': str(e)}
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring loop"""
        self.logger.info("Starting continuous enterprise monitoring...")
        
        interval = self.config['monitoring']['interval_seconds']
        
        try:
            while True:
                self.run_monitoring_cycle()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Continuous monitoring error: {e}")

def main():
    """
    Main monitoring function
    """
    print("🏢 Enterprise Monitoring System")
    print("=" * 50)
    
    monitor = EnterpriseMonitor()
    
    print("📊 Choose monitoring mode:")
    print("1. Single monitoring cycle")
    print("2. Continuous monitoring")
    print("3. Generate business intelligence report")
    print("4. View recent alerts")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🔍 Running single monitoring cycle...")
        result = monitor.run_monitoring_cycle()
        print(json.dumps(result, indent=2))
        
    elif choice == "2":
        print(f"\n🔄 Starting continuous monitoring (interval: {monitor.config['monitoring']['interval_seconds']}s)")
        print("Press Ctrl+C to stop")
        monitor.start_continuous_monitoring()
        
    elif choice == "3":
        print("\n📈 Generating business intelligence report...")
        report = monitor.generate_business_intelligence_report()
        print(json.dumps(report, indent=2))
        
    elif choice == "4":
        print("\n🚨 Recent alerts:")
        # TODO: Implement alert viewing
        print("Feature coming soon...")
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 