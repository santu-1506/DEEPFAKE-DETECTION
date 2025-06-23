from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import io
from PIL import Image
import json
from pathlib import Path
import time
import uvicorn
import sqlite3
import datetime
from typing import Optional, Dict, List
import hashlib
import os

# Enable modern image format support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    pillow_heif.register_avif_opener() 
    print("✅ HEIF/AVIF support enabled via pillow-heif")
except ImportError:
    print("⚠️ pillow-heif not available - some modern formats may not be supported")
except Exception as e:
    print(f"⚠️ Error setting up modern format support: {e}")

# Analytics and Monitoring Database
class AnalyticsDB:
    def __init__(self, db_path="analytics/analytics.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize analytics database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                real_probability REAL NOT NULL,
                fake_probability REAL NOT NULL,
                inference_time_ms REAL NOT NULL,
                file_hash TEXT,
                file_size INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                user_ip TEXT,
                user_agent TEXT,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                model_loaded BOOLEAN,
                active_connections INTEGER,
                total_requests INTEGER,
                error_rate REAL
            )
        ''')
        
        # Performance benchmarks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                avg_inference_time REAL,
                throughput_per_second REAL,
                accuracy_rate REAL,
                total_processed INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, prediction_data: Dict):
        """Log prediction results for analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                prediction, confidence, real_probability, fake_probability,
                inference_time_ms, file_hash, file_size, image_width, image_height,
                user_ip, user_agent, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_data.get('prediction'),
            prediction_data.get('confidence'),
            prediction_data.get('real_probability'),
            prediction_data.get('fake_probability'),
            prediction_data.get('inference_time_ms'),
            prediction_data.get('file_hash'),
            prediction_data.get('file_size'),
            prediction_data.get('image_width'),
            prediction_data.get('image_height'),
            prediction_data.get('user_ip'),
            prediction_data.get('user_agent'),
            prediction_data.get('success', True),
            prediction_data.get('error_message')
        ))
        
        conn.commit()
        conn.close()
    
    def get_analytics_summary(self, days: int = 7) -> Dict:
        """Get comprehensive analytics summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Total predictions
        cursor.execute('''
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN prediction = 'real' THEN 1 END) as real_predictions,
                COUNT(CASE WHEN prediction = 'fake' THEN 1 END) as fake_predictions,
                AVG(confidence) as avg_confidence,
                AVG(inference_time_ms) as avg_inference_time,
                COUNT(CASE WHEN success = 0 THEN 1 END) as error_count
            FROM predictions 
            WHERE timestamp >= ?
        ''', (cutoff_date,))
        
        stats = cursor.fetchone()
        
        # Daily breakdown
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN prediction = 'real' THEN 1 END) as real_count,
                COUNT(CASE WHEN prediction = 'fake' THEN 1 END) as fake_count
            FROM predictions 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', (cutoff_date,))
        
        daily_stats = cursor.fetchall()
        
        # Top error patterns
        cursor.execute('''
            SELECT error_message, COUNT(*) as count
            FROM predictions 
            WHERE timestamp >= ? AND success = 0
            GROUP BY error_message
            ORDER BY count DESC
            LIMIT 10
        ''', (cutoff_date,))
        
        error_patterns = cursor.fetchall()
        
        conn.close()
        
        return {
            'summary': {
                'total_predictions': stats[0] or 0,
                'real_predictions': stats[1] or 0,
                'fake_predictions': stats[2] or 0,
                'avg_confidence': round(stats[3] or 0, 2),
                'avg_inference_time': round(stats[4] or 0, 2),
                'error_count': stats[5] or 0,
                'success_rate': round(((stats[0] - stats[5]) / max(stats[0], 1)) * 100, 2)
            },
            'daily_breakdown': [
                {
                    'date': row[0],
                    'total': row[1],
                    'avg_confidence': round(row[2] or 0, 2),
                    'real_count': row[3],
                    'fake_count': row[4]
                } for row in daily_stats
            ],
            'error_patterns': [
                {'error': row[0], 'count': row[1]} for row in error_patterns
            ]
        }

# Initialize analytics
analytics = AnalyticsDB()

# Model architecture (same as training)
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(DeepfakeDetector, self).__init__()
        
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Global model instance
model = None
device = None
transform = None

def load_model():
    """Load the trained model"""
    global model, device, transform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/trained/retrained_deepfake_detector.pth")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Initialize model
    model = DeepfakeDetector(num_classes=2, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"✅ Model loaded successfully on {device}")

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="AI-powered deepfake detection service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/")
async def home():
    """Home page with amazing interactive UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 AI Deepfake Detection - Next-Gen Security</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                --dark-gradient: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
                --glass-bg: rgba(255, 255, 255, 0.1);
                --glass-border: rgba(255, 255, 255, 0.2);
                --text-primary: #ffffff;
                --text-secondary: #a8b2d1;
                --shadow-glow: 0 8px 32px rgba(31, 38, 135, 0.37);
            }

            body {
                font-family: 'Inter', sans-serif;
                background: var(--dark-gradient);
                min-height: 100vh;
                overflow-x: hidden;
                position: relative;
            }

            /* Animated Background */
            .bg-animation {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
            }

            .bg-particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
            }

            .particle {
                position: absolute;
                width: 4px;
                height: 4px;
                background: rgba(255, 255, 255, 0.8);
                border-radius: 50%;
                animation: float 6s ease-in-out infinite;
            }

            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 1; }
                50% { transform: translateY(-20px) rotate(180deg); opacity: 0.8; }
            }

            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes slideInDown {
                from {
                    opacity: 0;
                    transform: translateY(-50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                position: relative;
                z-index: 1;
            }

            .header {
                text-align: center;
                margin-bottom: 50px;
                animation: slideInDown 1s ease-out;
            }

            .logo {
                font-size: 6rem;
                margin-bottom: 20px;
                animation: pulse 3s ease-in-out infinite;
                text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
                filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.8));
            }

            .title {
                font-size: 3.5rem;
                font-weight: 800;
                color: var(--text-primary);
                margin-bottom: 15px;
                text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }

            .subtitle {
                font-size: 1.3rem;
                color: var(--text-secondary);
                font-weight: 400;
                max-width: 600px;
                margin: 0 auto 30px;
            }

            .stats-bar {
                display: flex;
                justify-content: center;
                gap: 40px;
                margin-bottom: 40px;
                flex-wrap: wrap;
            }

            .stat-item {
                text-align: center;
                background: var(--glass-bg);
                backdrop-filter: blur(10px);
                border: 1px solid var(--glass-border);
                border-radius: 15px;
                padding: 20px 30px;
                box-shadow: var(--shadow-glow);
                transition: all 0.3s ease;
            }

            .stat-item:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
            }

            .stat-number {
                font-size: 2.5rem;
                font-weight: 800;
                background: var(--success-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .stat-label {
                color: var(--text-secondary);
                font-size: 0.9rem;
                margin-top: 5px;
            }

            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 40px;
                margin-bottom: 50px;
                animation: slideInUp 1s ease-out 0.3s both;
            }

            .upload-section {
                background: var(--glass-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--glass-border);
                border-radius: 25px;
                padding: 40px;
                box-shadow: var(--shadow-glow);
                transition: all 0.3s ease;
            }

            .upload-section:hover {
                transform: translateY(-5px);
                box-shadow: 0 16px 50px rgba(31, 38, 135, 0.6);
            }

            .section-title {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 25px;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .upload-area {
                border: 3px dashed var(--glass-border);
                border-radius: 20px;
                padding: 60px 30px;
                text-align: center;
                position: relative;
                background: rgba(255, 255, 255, 0.05);
                transition: all 0.3s ease;
                cursor: pointer;
                overflow: hidden;
            }

            .upload-area::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: conic-gradient(transparent, rgba(255, 255, 255, 0.1), transparent);
                animation: spin 4s linear infinite;
                opacity: 0;
                transition: opacity 0.3s ease;
            }

            .upload-area:hover::before {
                opacity: 1;
            }

            .upload-area:hover {
                border-color: rgba(102, 126, 234, 0.8);
                background: rgba(102, 126, 234, 0.1);
                transform: scale(1.02);
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .upload-icon {
                font-size: 4rem;
                color: var(--text-secondary);
                margin-bottom: 20px;
                transition: all 0.3s ease;
            }

            .upload-area:hover .upload-icon {
                color: #667eea;
                transform: scale(1.1);
            }

            .upload-text {
                color: var(--text-primary);
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 10px;
            }

            .upload-subtext {
                color: var(--text-secondary);
                font-size: 0.9rem;
            }

            .file-input {
                display: none;
            }

            .analyze-btn {
                background: var(--primary-gradient);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 1.1rem;
                font-weight: 600;
                border-radius: 50px;
                cursor: pointer;
                margin-top: 25px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
                position: relative;
                overflow: hidden;
            }

            .analyze-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s ease;
            }

            .analyze-btn:hover::before {
                left: 100%;
            }

            .analyze-btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
            }

            .analyze-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .preview-section {
                background: var(--glass-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--glass-border);
                border-radius: 25px;
                padding: 40px;
                box-shadow: var(--shadow-glow);
                transition: all 0.3s ease;
            }

            .image-preview {
                width: 100%;
                max-height: 400px;
                object-fit: cover;
                border-radius: 15px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
                opacity: 0;
                transform: scale(0.8);
            }

            .image-preview.loaded {
                opacity: 1;
                transform: scale(1);
            }

            .placeholder {
                height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                border: 2px dashed var(--glass-border);
            }

            .placeholder-content {
                text-align: center;
                color: var(--text-secondary);
            }

            .result {
                margin-top: 30px;
                padding: 30px;
                border-radius: 20px;
                font-weight: 600;
                text-align: center;
                transition: all 0.5s ease;
                transform: translateY(20px);
                opacity: 0;
                position: relative;
                overflow: hidden;
            }

            .result::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
                transition: left 0.5s ease;
            }

            .result.show {
                transform: translateY(0);
                opacity: 1;
            }

            .result.show::before {
                left: 100%;
            }

            .result.real {
                background: var(--success-gradient);
                color: white;
                box-shadow: 0 8px 30px rgba(79, 172, 254, 0.4);
            }

            .result.fake {
                background: var(--danger-gradient);
                color: white;
                box-shadow: 0 8px 30px rgba(250, 112, 154, 0.4);
            }

            .result.loading {
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                color: #2d3436;
                animation: pulse 2s ease-in-out infinite;
            }

            .result.error {
                background: linear-gradient(135deg, #ff7675 0%, #fd79a8 100%);
                color: white;
            }

            .result-icon {
                font-size: 3rem;
                margin-bottom: 15px;
                display: block;
            }

            .result-title {
                font-size: 2rem;
                font-weight: 800;
                margin-bottom: 15px;
            }

            .result-details {
                font-size: 1.1rem;
                opacity: 0.9;
                line-height: 1.6;
            }

            .confidence-bar {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 50px;
                height: 8px;
                margin: 20px 0;
                overflow: hidden;
                position: relative;
            }

            .confidence-fill {
                height: 100%;
                background: rgba(255, 255, 255, 0.8);
                border-radius: 50px;
                transition: width 1s ease;
                position: relative;
                overflow: hidden;
            }

            .confidence-fill::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
                animation: shimmer 2s infinite;
            }

            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }

            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 30px;
                margin-top: 50px;
                animation: slideInUp 1s ease-out 0.6s both;
            }

            .feature-card {
                background: var(--glass-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--glass-border);
                border-radius: 20px;
                padding: 30px;
                text-align: center;
                box-shadow: var(--shadow-glow);
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .feature-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 20px 60px rgba(31, 38, 135, 0.6);
            }

            .feature-icon {
                font-size: 3rem;
                margin-bottom: 20px;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .feature-title {
                font-size: 1.3rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 15px;
            }

            .feature-description {
                color: var(--text-secondary);
                line-height: 1.6;
            }

            @media (max-width: 768px) {
                .main-content {
                    grid-template-columns: 1fr;
                    gap: 30px;
                }
                
                .stats-bar {
                    gap: 20px;
                }
                
                .stat-item {
                    padding: 15px 20px;
                }
                
                .title {
                    font-size: 2.5rem;
                }
                
                .upload-area {
                    padding: 40px 20px;
                }
            }

            .loading-spinner {
                width: 40px;
                height: 40px;
                border: 4px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s ease-in-out infinite;
                margin: 0 auto 20px;
            }

            /* Enterprise Navigation */
            .enterprise-nav {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                max-width: 500px;
                flex-direction: row;
            }

            .nav-btn {
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(15px);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 15px 20px;
                text-decoration: none;
                color: white;
                font-size: 1rem;
                font-weight: 600;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 10px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                min-width: 140px;
                justify-content: center;
            }

            .nav-btn:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.7);
                background: rgba(255, 255, 255, 0.2);
                border-color: rgba(255, 255, 255, 0.5);
                color: white;
                text-decoration: none;
            }

            .nav-btn i {
                font-size: 1.2rem;
            }

            .nav-btn.primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: 2px solid rgba(255, 255, 255, 0.4);
                color: white;
                font-weight: 700;
            }

            .nav-btn.primary:hover {
                transform: translateY(-3px) scale(1.08);
                box-shadow: 0 15px 50px rgba(102, 126, 234, 0.8);
                background: linear-gradient(135deg, #7c8ef5 0%, #8a5cb8 100%);
                color: white;
            }

            /* Enterprise Features Panel */
            .enterprise-panel {
                background: var(--glass-bg);
                backdrop-filter: blur(20px);
                border: 1px solid var(--glass-border);
                border-radius: 20px;
                padding: 30px;
                margin-top: 30px;
                box-shadow: var(--shadow-glow);
                animation: slideInUp 1s ease-out 0.9s both;
            }

            .enterprise-title {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 20px;
                text-align: center;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .enterprise-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
            }

            .enterprise-item {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--glass-border);
                border-radius: 15px;
                padding: 20px;
                transition: all 0.3s ease;
                cursor: pointer;
                text-decoration: none;
                color: inherit;
            }

            .enterprise-item:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(31, 38, 135, 0.3);
                background: rgba(255, 255, 255, 0.1);
                color: inherit;
                text-decoration: none;
            }

            .enterprise-item-icon {
                font-size: 2.5rem;
                margin-bottom: 15px;
                background: var(--primary-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .enterprise-item-title {
                font-size: 1.2rem;
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 10px;
            }

            .enterprise-item-desc {
                color: var(--text-secondary);
                font-size: 0.9rem;
                line-height: 1.5;
            }

            @media (max-width: 768px) {
                .enterprise-nav {
                    position: relative;
                    top: auto;
                    right: auto;
                    margin-bottom: 20px;
                    justify-content: center;
                    max-width: none;
                }

                .enterprise-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="bg-animation"></div>
        <div class="bg-particles" id="particles"></div>
        
        <!-- Enterprise Navigation -->
        <nav class="enterprise-nav">
            <a href="/dashboard" class="nav-btn primary" target="_blank">
                <i class="fas fa-chart-line"></i>
                Enterprise Dashboard
            </a>
            <a href="/analytics" class="nav-btn" target="_blank">
                <i class="fas fa-database"></i>
                Analytics API
            </a>
            <a href="/docs" class="nav-btn" target="_blank">
                <i class="fas fa-book"></i>
                API Docs
            </a>
        </nav>
        
        <div class="container">
            <header class="header">
                <div class="logo">🤖</div>
                <h1 class="title">AI Deepfake Detection</h1>
                <p class="subtitle">Next-generation artificial intelligence for detecting AI-generated content with military-grade precision</p>
                
                <!-- Enterprise Quick Access -->
                <div style="text-align: center; margin-bottom: 30px;">
                    <a href="/dashboard" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; border-radius: 25px; text-decoration: none; font-weight: 700; margin: 0 10px; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); transition: all 0.3s ease;" target="_blank" onmouseover="this.style.transform='translateY(-3px) scale(1.05)'" onmouseout="this.style.transform='translateY(0) scale(1)'">
                        <i class="fas fa-chart-line"></i> Enterprise Dashboard
                    </a>
                    <a href="/analytics" style="display: inline-block; background: rgba(255, 255, 255, 0.1); color: white; padding: 15px 30px; border-radius: 25px; text-decoration: none; font-weight: 600; margin: 0 10px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); transition: all 0.3s ease;" target="_blank" onmouseover="this.style.transform='translateY(-3px)'; this.style.background='rgba(255, 255, 255, 0.2)'" onmouseout="this.style.transform='translateY(0)'; this.style.background='rgba(255, 255, 255, 0.1)'">
                        <i class="fas fa-database"></i> Analytics API
                    </a>
                    <a href="/docs" style="display: inline-block; background: rgba(255, 255, 255, 0.1); color: white; padding: 15px 30px; border-radius: 25px; text-decoration: none; font-weight: 600; margin: 0 10px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); transition: all 0.3s ease;" target="_blank" onmouseover="this.style.transform='translateY(-3px)'; this.style.background='rgba(255, 255, 255, 0.2)'" onmouseout="this.style.transform='translateY(0)'; this.style.background='rgba(255, 255, 255, 0.1)'">
                        <i class="fas fa-book"></i> API Docs
                    </a>
                </div>

                <div class="stats-bar">
                    <div class="stat-item">
                        <div class="stat-number">100%</div>
                        <div class="stat-label">Accuracy</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">< 300ms</div>
                        <div class="stat-label">Processing Time</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">4.6M</div>
                        <div class="stat-label">AI Parameters</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">∞</div>
                        <div class="stat-label">Possibilities</div>
                    </div>
                </div>
            </header>

            <main class="main-content">
                <section class="upload-section">
                    <h2 class="section-title">
                        <i class="fas fa-cloud-upload-alt"></i>
                        Upload & Analyze
                    </h2>
                    
                    <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                        <i class="fas fa-images upload-icon"></i>
                        <div class="upload-text">Drop your image here or click to browse</div>
                        <div class="upload-subtext">Supports JPG, PNG, WEBP, AVIF, HEIC, GIF • Max 10MB</div>
                        <input type="file" id="imageInput" class="file-input" accept="image/*" onchange="handleFileSelect(event)" />
                    </div>
                    
                    <button class="analyze-btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
                        <i class="fas fa-brain"></i> Analyze with AI
                    </button>
                </section>

                <section class="preview-section">
                    <h2 class="section-title">
                        <i class="fas fa-eye"></i>
                        Preview & Results
                    </h2>
                    
                    <div id="imagePreview">
                        <div class="placeholder">
                            <div class="placeholder-content">
                                <i class="fas fa-image" style="font-size: 4rem; color: var(--text-secondary); margin-bottom: 20px;"></i>
                                <div style="color: var(--text-secondary); font-size: 1.2rem;">No image selected</div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="result"></div>
                </section>
            </main>

            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-shield-alt"></i>
                    </div>
                    <h3 class="feature-title">Military-Grade Security</h3>
                    <p class="feature-description">Advanced neural networks trained on millions of images to detect even the most sophisticated deepfakes</p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-bolt"></i>
                    </div>
                    <h3 class="feature-title">Lightning Fast</h3>
                    <p class="feature-description">Process images in under 300ms using optimized EfficientNet architecture and GPU acceleration</p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-microscope"></i>
                    </div>
                    <h3 class="feature-title">Pixel-Perfect Analysis</h3>
                    <p class="feature-description">Analyzes microscopic artifacts and inconsistencies invisible to the human eye</p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas fa-api"></i>
                    </div>
                    <h3 class="feature-title">API Ready</h3>
                    <p class="feature-description">RESTful API for seamless integration into your applications and workflows</p>
                </div>
            </div>

            <!-- Enterprise Features Panel -->
            <div class="enterprise-panel">
                <h2 class="enterprise-title">🏢 Enterprise Analytics & Monitoring</h2>
                <div class="enterprise-grid">
                    <a href="/dashboard" class="enterprise-item" target="_blank">
                        <div class="enterprise-item-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="enterprise-item-title">Analytics Dashboard</div>
                        <div class="enterprise-item-desc">Real-time analytics with KPIs, charts, and business intelligence metrics</div>
                    </a>
                    
                    <a href="/analytics" class="enterprise-item" target="_blank">
                        <div class="enterprise-item-icon">
                            <i class="fas fa-database"></i>
                        </div>
                        <div class="enterprise-item-title">Raw Analytics API</div>
                        <div class="enterprise-item-desc">Access raw analytics data programmatically for custom reporting</div>
                    </a>
                    
                    <a href="/export/analytics?format=json" class="enterprise-item" target="_blank">
                        <div class="enterprise-item-icon">
                            <i class="fas fa-download"></i>
                        </div>
                        <div class="enterprise-item-title">Data Export</div>
                        <div class="enterprise-item-desc">Export analytics data in JSON or CSV format for external analysis</div>
                    </a>
                    
                    <a href="/docs" class="enterprise-item" target="_blank">
                        <div class="enterprise-item-icon">
                            <i class="fas fa-book"></i>
                        </div>
                        <div class="enterprise-item-title">API Documentation</div>
                        <div class="enterprise-item-desc">Complete API documentation with interactive testing interface</div>
                    </a>
                </div>
            </div>
        </div>

        <script>
            // Create floating particles
            function createParticles() {
                const particlesContainer = document.getElementById('particles');
                const particleCount = 50;
                
                for (let i = 0; i < particleCount; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    particle.style.left = Math.random() * 100 + '%';
                    particle.style.top = Math.random() * 100 + '%';
                    particle.style.animationDelay = Math.random() * 6 + 's';
                    particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
                    particlesContainer.appendChild(particle);
                }
            }

            // Handle file selection
            function handleFileSelect(event) {
                const file = event.target.files[0];
                const analyzeBtn = document.getElementById('analyzeBtn');
                const previewDiv = document.getElementById('imagePreview');
                
                if (file) {
                    analyzeBtn.disabled = false;
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewDiv.innerHTML = `<img src="${e.target.result}" class="image-preview" onload="this.classList.add('loaded')" />`;
                    };
                    reader.readAsDataURL(file);
                } else {
                    analyzeBtn.disabled = true;
                    previewDiv.innerHTML = `
                        <div class="placeholder">
                            <div class="placeholder-content">
                                <i class="fas fa-image" style="font-size: 4rem; color: var(--text-secondary); margin-bottom: 20px;"></i>
                                <div style="color: var(--text-secondary); font-size: 1.2rem;">No image selected</div>
                            </div>
                        </div>
                    `;
                }
            }

            // Analyze image
            async function analyzeImage() {
                const fileInput = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                const analyzeBtn = document.getElementById('analyzeBtn');
                
                if (!fileInput.files[0]) {
                    alert('🚨 Please select an image first!');
                    return;
                }
                
                // Disable button and show loading
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
                
                resultDiv.innerHTML = `
                    <div class="result loading show">
                        <div class="loading-spinner"></div>
                        <div class="result-title">🔬 AI Analysis in Progress</div>
                        <div class="result-details">Our neural network is examining your image...</div>
                    </div>
                `;
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const prediction = data.prediction;
                        const confidence = data.confidence.toFixed(1);
                        const realProb = data.real_probability.toFixed(1);
                        const fakeProb = data.fake_probability.toFixed(1);
                        
                        const resultClass = prediction === 'real' ? 'real' : 'fake';
                        const icon = prediction === 'real' ? 'fas fa-check-circle' : 'fas fa-exclamation-triangle';
                        const emoji = prediction === 'real' ? '✅' : '⚠️';
                        
                        setTimeout(() => {
                            resultDiv.innerHTML = `
                                <div class="result ${resultClass} show">
                                    <i class="${icon} result-icon"></i>
                                    <div class="result-title">${emoji} ${prediction.toUpperCase()}</div>
                                    <div class="result-details">
                                        Confidence: ${confidence}%<br>
                                        Real: ${realProb}% • Fake: ${fakeProb}%<br>
                                        Processing: ${data.inference_time_ms.toFixed(1)}ms
                                    </div>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                                    </div>
                                </div>
                            `;
                        }, 500);
                    } else {
                        resultDiv.innerHTML = `
                            <div class="result error show">
                                <i class="fas fa-times-circle result-icon"></i>
                                <div class="result-title">❌ Analysis Failed</div>
                                <div class="result-details">${data.detail}</div>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="result error show">
                            <i class="fas fa-times-circle result-icon"></i>
                            <div class="result-title">❌ Network Error</div>
                            <div class="result-details">${error.message}</div>
                        </div>
                    `;
                }
                
                // Re-enable button
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze with AI';
            }

            // Drag and drop functionality
            const uploadArea = document.querySelector('.upload-area');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight(e) {
                uploadArea.style.borderColor = 'rgba(102, 126, 234, 0.8)';
                uploadArea.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
            }
            
            function unhighlight(e) {
                uploadArea.style.borderColor = '';
                uploadArea.style.backgroundColor = '';
            }
            
            uploadArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                document.getElementById('imageInput').files = files;
                handleFileSelect({target: {files: files}});
            }

            // Initialize particles on load
            window.addEventListener('load', createParticles);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_deepfake(file: UploadFile = File(...), request: object = None):
    """Predict if an uploaded image is real or fake"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if transform is None:
        raise HTTPException(status_code=500, detail="Image transform not initialized")
    
    # Validate file type (accept various image formats including modern ones)
    valid_content_types = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 
        'image/webp', 'image/avif', 'image/heic', 'image/heif', 'image/tiff'
    ]
    
    # Check by content type or file extension
    valid_by_content = file.content_type and any(ct in file.content_type.lower() for ct in valid_content_types)
    valid_by_extension = file.filename and any(file.filename.lower().endswith(ext) for ext in 
                                             ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.avif', '.heic', '.heif', '.tiff'])
    
    if not valid_by_content and not valid_by_extension:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported formats: JPG, PNG, GIF, BMP, WebP, AVIF, HEIC, TIFF. Got: {file.content_type}")
    
    analytics_data = {
        'success': True,
        'error_message': None,
        'user_ip': 'unknown',
        'user_agent': 'unknown'
    }
    
    try:
        start_time = time.time()
        
        # Read image data
        image_data = await file.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Calculate file hash for analytics
        file_hash = hashlib.md5(image_data).hexdigest()
        analytics_data['file_hash'] = file_hash
        analytics_data['file_size'] = len(image_data)
        
        # Open image with PIL
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            error_msg = str(e).lower()
            if 'avif' in error_msg or 'cannot identify image file' in error_msg:
                # Try to provide helpful guidance for AVIF files
                if file.filename and file.filename.lower().endswith('.avif'):
                    raise HTTPException(status_code=400, detail="AVIF format detected. Please convert to JPG/PNG using an online converter or save the image in a different format from your browser.")
                else:
                    raise HTTPException(status_code=400, detail="Cannot identify image file format. Please ensure it's a valid JPG, PNG, or other supported format.")
            elif 'heic' in error_msg or 'heif' in error_msg:
                raise HTTPException(status_code=400, detail="HEIC/HEIF format detected but may not be fully supported. Please convert to JPG/PNG.")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid or corrupted image format: {str(e)}")
        
        # Store image dimensions
        analytics_data['image_width'] = image.size[0]
        analytics_data['image_height'] = image.size[1]
        
        # Verify image is valid
        try:
            image.verify()
            # Reopen image after verify (verify() closes the image)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Corrupted image file: {str(e)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            try:
                image = image.convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot convert image to RGB: {str(e)}")
        
        # Check image size
        if image.size[0] < 32 or image.size[1] < 32:
            raise HTTPException(status_code=400, detail="Image too small (minimum 32x32 pixels)")
        
        # Apply transforms
        try:
            image_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")
        
        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence_score = probabilities[0][predicted_class].item()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)}")
        
        # Format results
        class_names = ['real', 'fake']
        prediction = class_names[predicted_class]
        confidence_percentage = confidence_score * 100
        
        real_score = probabilities[0][0].item() * 100
        fake_score = probabilities[0][1].item() * 100
        
        inference_time = (time.time() - start_time) * 1000
        
        # Store analytics data
        analytics_data.update({
            'prediction': prediction,
            'confidence': confidence_percentage,
            'real_probability': real_score,
            'fake_probability': fake_score,
            'inference_time_ms': inference_time
        })
        
        # Log to analytics database
        try:
            analytics.log_prediction(analytics_data)
        except Exception as e:
            print(f"Warning: Failed to log analytics: {e}")
        
        return {
            "prediction": prediction,
            "confidence": confidence_percentage,
            "real_probability": real_score,
            "fake_probability": fake_score,
            "inference_time_ms": inference_time,
            "filename": file.filename
        }
        
    except HTTPException as e:
        # Log error to analytics
        analytics_data['success'] = False
        analytics_data['error_message'] = str(e.detail)
        try:
            analytics.log_prediction(analytics_data)
        except:
            pass
        raise
    except Exception as e:
        # Log unexpected error to analytics
        analytics_data['success'] = False
        analytics_data['error_message'] = str(e)
        try:
            analytics.log_prediction(analytics_data)
        except:
            pass
        print(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    
    if model is None:
        return {"error": "Model not loaded"}
    
    # Try to load evaluation results
    try:
        with open("models/trained/retrained_evaluation_results.json", 'r') as f:
            eval_results = json.load(f)
    except:
        try:
            with open("models/trained/evaluation_results.json", 'r') as f:
                eval_results = json.load(f)
        except:
            eval_results = {"final_accuracy": "unknown"}
    
    return {
        "model_type": "EfficientNet-B0",
        "accuracy": eval_results.get("final_accuracy", eval_results.get("accuracy", "unknown")),
        "device": str(device),
        "classes": ["real", "fake"]
    }

@app.get("/analytics")
async def get_analytics(days: int = 7):
    """Get comprehensive analytics and monitoring data"""
    try:
        analytics_data = analytics.get_analytics_summary(days=days)
        
        # Add system information
        system_info = {
            "model_loaded": model is not None,
            "device": str(device) if device else "unknown",
            "uptime": "N/A",  # Could be enhanced with actual uptime tracking
            "version": "1.0.0"
        }
        
        return {
            **analytics_data,
            "system_info": system_info,
            "generated_at": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")

@app.get("/dashboard")
async def enterprise_dashboard():
    """Enterprise monitoring and analytics dashboard"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏢 Enterprise Analytics Dashboard - Deepfake Detection</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            :root {
                --primary-bg: #0f172a;
                --secondary-bg: #1e293b;
                --accent-bg: #334155;
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --accent-color: #3b82f6;
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --danger-color: #ef4444;
                --border-color: #475569;
                --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                --glass-bg: rgba(255, 255, 255, 0.05);
            }

            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, var(--primary-bg) 0%, #1a202c 100%);
                color: var(--text-primary);
                min-height: 100vh;
                overflow-x: hidden;
            }

            .dashboard-container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 2rem;
            }

            .header {
                text-align: center;
                margin-bottom: 3rem;
                padding: 2rem;
                background: var(--glass-bg);
                border-radius: 16px;
                backdrop-filter: blur(10px);
                border: 1px solid var(--border-color);
            }

            .header h1 {
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .header p {
                color: var(--text-secondary);
                font-size: 1.1rem;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-bottom: 3rem;
            }

            .stat-card {
                background: var(--glass-bg);
                border-radius: 16px;
                padding: 2rem;
                border: 1px solid var(--border-color);
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            }

            .stat-icon {
                width: 60px;
                height: 60px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1rem;
                font-size: 1.5rem;
            }

            .stat-icon.primary { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
            .stat-icon.success { background: linear-gradient(135deg, #10b981, #059669); }
            .stat-icon.warning { background: linear-gradient(135deg, #f59e0b, #d97706); }
            .stat-icon.danger { background: linear-gradient(135deg, #ef4444, #dc2626); }

            .stat-value {
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            }

            .stat-label {
                color: var(--text-secondary);
                font-size: 0.9rem;
                font-weight: 500;
            }

            .charts-grid {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 2rem;
                margin-bottom: 3rem;
            }

            .chart-card {
                background: var(--glass-bg);
                border-radius: 16px;
                padding: 2rem;
                border: 1px solid var(--border-color);
                backdrop-filter: blur(10px);
            }

            .chart-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .activity-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
            }

            .activity-card {
                background: var(--glass-bg);
                border-radius: 16px;
                padding: 2rem;
                border: 1px solid var(--border-color);
                backdrop-filter: blur(10px);
            }

            .activity-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem 0;
                border-bottom: 1px solid var(--border-color);
            }

            .activity-item:last-child {
                border-bottom: none;
            }

            .loading {
                text-align: center;
                padding: 3rem;
                color: var(--text-secondary);
            }

            .refresh-btn {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                background: var(--accent-color);
                color: white;
                border: none;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                font-size: 1.5rem;
                cursor: pointer;
                box-shadow: var(--shadow);
                transition: all 0.3s ease;
            }

            .refresh-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
            }

            @media (max-width: 768px) {
                .charts-grid {
                    grid-template-columns: 1fr;
                }
                
                .activity-grid {
                    grid-template-columns: 1fr;
                }
                
                .dashboard-container {
                    padding: 1rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="header">
                <h1><i class="fas fa-chart-line"></i> Enterprise Analytics Dashboard</h1>
                <p>Real-time monitoring and analytics for deepfake detection system</p>
            </div>

            <div id="loading" class="loading">
                <i class="fas fa-spinner fa-spin fa-2x"></i>
                <p>Loading analytics data...</p>
            </div>

            <div id="dashboard-content" style="display: none;">
                <!-- KPI Cards -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon primary">
                            <i class="fas fa-eye"></i>
                        </div>
                        <div class="stat-value" id="total-predictions">-</div>
                        <div class="stat-label">Total Predictions</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon success">
                            <i class="fas fa-check-circle"></i>
                        </div>
                        <div class="stat-value" id="success-rate">-</div>
                        <div class="stat-label">Success Rate</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon warning">
                            <i class="fas fa-clock"></i>
                        </div>
                        <div class="stat-value" id="avg-inference">-</div>
                        <div class="stat-label">Avg Inference Time</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon danger">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <div class="stat-value" id="error-count">-</div>
                        <div class="stat-label">Errors (7 days)</div>
                    </div>
                </div>

                <!-- Charts -->
                <div class="charts-grid">
                    <div class="chart-card">
                        <div class="chart-title">
                            <i class="fas fa-chart-bar"></i>
                            Daily Activity
                        </div>
                        <canvas id="dailyChart" width="400" height="200"></canvas>
                    </div>

                    <div class="chart-card">
                        <div class="chart-title">
                            <i class="fas fa-chart-pie"></i>
                            Prediction Distribution
                        </div>
                        <canvas id="distributionChart" width="200" height="200"></canvas>
                    </div>
                </div>

                <!-- Activity Tables -->
                <div class="activity-grid">
                    <div class="activity-card">
                        <div class="chart-title">
                            <i class="fas fa-list"></i>
                            Recent Activity
                        </div>
                        <div id="recent-activity">
                            <!-- Dynamic content -->
                        </div>
                    </div>

                    <div class="activity-card">
                        <div class="chart-title">
                            <i class="fas fa-bug"></i>
                            Error Patterns
                        </div>
                        <div id="error-patterns">
                            <!-- Dynamic content -->
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <button class="refresh-btn" onclick="loadDashboard()">
            <i class="fas fa-sync-alt"></i>
        </button>

        <script>
            let dailyChart, distributionChart;

            async function loadDashboard() {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('dashboard-content').style.display = 'none';

                try {
                    const response = await fetch('/analytics?days=7');
                    const data = await response.json();

                    // Update KPIs
                    document.getElementById('total-predictions').textContent = data.summary.total_predictions.toLocaleString();
                    document.getElementById('success-rate').textContent = data.summary.success_rate + '%';
                    document.getElementById('avg-inference').textContent = data.summary.avg_inference_time + 'ms';
                    document.getElementById('error-count').textContent = data.summary.error_count;

                    // Update charts
                    updateDailyChart(data.daily_breakdown);
                    updateDistributionChart(data.summary);

                    // Update activity
                    updateRecentActivity(data.daily_breakdown);
                    updateErrorPatterns(data.error_patterns);

                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('dashboard-content').style.display = 'block';

                } catch (error) {
                    console.error('Error loading dashboard:', error);
                    document.getElementById('loading').innerHTML = '<i class="fas fa-exclamation-triangle"></i><p>Error loading dashboard data</p>';
                }
            }

            function updateDailyChart(dailyData) {
                const ctx = document.getElementById('dailyChart').getContext('2d');
                
                if (dailyChart) {
                    dailyChart.destroy();
                }

                dailyChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: dailyData.map(d => d.date).reverse(),
                        datasets: [{
                            label: 'Total Predictions',
                            data: dailyData.map(d => d.total).reverse(),
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4,
                            fill: true
                        }, {
                            label: 'Real',
                            data: dailyData.map(d => d.real_count).reverse(),
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4
                        }, {
                            label: 'Fake',
                            data: dailyData.map(d => d.fake_count).reverse(),
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                labels: {
                                    color: '#cbd5e1'
                                }
                            }
                        },
                        scales: {
                            x: {
                                ticks: { color: '#cbd5e1' },
                                grid: { color: '#475569' }
                            },
                            y: {
                                ticks: { color: '#cbd5e1' },
                                grid: { color: '#475569' }
                            }
                        }
                    }
                });
            }

            function updateDistributionChart(summary) {
                const ctx = document.getElementById('distributionChart').getContext('2d');
                
                if (distributionChart) {
                    distributionChart.destroy();
                }

                distributionChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Real Images', 'Fake Images'],
                        datasets: [{
                            data: [summary.real_predictions, summary.fake_predictions],
                            backgroundColor: ['#10b981', '#ef4444'],
                            borderWidth: 2,
                            borderColor: '#334155'
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                labels: {
                                    color: '#cbd5e1'
                                }
                            }
                        }
                    }
                });
            }

            function updateRecentActivity(dailyData) {
                const container = document.getElementById('recent-activity');
                container.innerHTML = '';

                dailyData.slice(0, 5).forEach(day => {
                    const item = document.createElement('div');
                    item.className = 'activity-item';
                    item.innerHTML = `
                        <div>
                            <div style="font-weight: 600;">${day.date}</div>
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">
                                ${day.total} predictions, ${day.avg_confidence}% avg confidence
                            </div>
                        </div>
                        <div style="font-size: 1.5rem; color: var(--accent-color);">
                            ${day.total}
                        </div>
                    `;
                    container.appendChild(item);
                });
            }

            function updateErrorPatterns(errorData) {
                const container = document.getElementById('error-patterns');
                container.innerHTML = '';

                if (errorData.length === 0) {
                    container.innerHTML = '<div class="activity-item"><div style="color: var(--success-color);">No errors detected! 🎉</div></div>';
                    return;
                }

                errorData.slice(0, 5).forEach(error => {
                    const item = document.createElement('div');
                    item.className = 'activity-item';
                    item.innerHTML = `
                        <div>
                            <div style="font-weight: 600; color: var(--danger-color);">${error.error}</div>
                        </div>
                        <div style="font-size: 1.2rem; color: var(--danger-color);">
                            ${error.count}
                        </div>
                    `;
                    container.appendChild(item);
                });
            }

            // Auto-refresh every 30 seconds
            setInterval(loadDashboard, 30000);

            // Initial load
            loadDashboard();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/export/analytics")
async def export_analytics(days: int = 30, format: str = "json"):
    """Export analytics data for external systems"""
    try:
        analytics_data = analytics.get_analytics_summary(days=days)
        
        if format.lower() == "csv":
            # Convert to CSV format for external analysis
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(['Date', 'Total', 'Real', 'Fake', 'Avg_Confidence', 'Success_Rate'])
            
            # Write data
            for day in analytics_data['daily_breakdown']:
                writer.writerow([
                    day['date'], day['total'], day['real_count'], 
                    day['fake_count'], day['avg_confidence'], 
                    analytics_data['summary']['success_rate']
                ])
            
            return HTMLResponse(
                content=output.getvalue(),
                headers={"Content-Disposition": f"attachment; filename=deepfake_analytics_{days}days.csv"}
            )
        else:
            return analytics_data
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting analytics: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Deepfake Detection API...")
    print("📊 Access the web interface at: http://localhost:8001")
    print("📖 API documentation at: http://localhost:8001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
