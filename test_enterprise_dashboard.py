#!/usr/bin/env python3
"""
Enterprise Dashboard Demo Script
================================

This script demonstrates the enterprise analytics and monitoring capabilities
by simulating API usage and showing how the dashboard collects and displays
comprehensive business intelligence data.
"""

import requests
import json
import time
import random
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

class DashboardDemo:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.db_path = "analytics/analytics.db"
    
    def check_server_status(self):
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running!")
                return True
            else:
                print(f"❌ API server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API server: {e}")
            print("🔧 Please start the server with: python api/main.py")
            return False
    
    def simulate_api_usage(self, num_requests=10):
        """Simulate API usage to generate analytics data"""
        print(f"\n🔄 Simulating {num_requests} API requests...")
        
        # Use existing test images from the processed folder
        test_images = []
        
        # Real images
        real_path = Path("data/processed/real")
        if real_path.exists():
            test_images.extend([(f, "real") for f in list(real_path.glob("*.jpg"))[:5]])
        
        # Fake images
        fake_path = Path("data/processed/fake")
        if fake_path.exists():
            test_images.extend([(f, "fake") for f in list(fake_path.glob("*.jpg"))[:5]])
        
        if not test_images:
            print("❌ No test images found in data/processed/")
            return
        
        success_count = 0
        for i in range(num_requests):
            try:
                # Randomly select an image
                image_path, expected_type = random.choice(test_images)
                
                with open(image_path, 'rb') as f:
                    files = {'file': (image_path.name, f, 'image/jpeg')}
                    response = requests.post(f"{self.api_url}/predict", files=files, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get('prediction', 'unknown')
                    confidence = result.get('confidence', 0)
                    inference_time = result.get('inference_time_ms', 0)
                    
                    success_count += 1
                    print(f"  {i+1:2d}. {image_path.name}: {prediction} ({confidence:.1f}%) - {inference_time:.1f}ms")
                else:
                    print(f"  {i+1:2d}. Error: {response.status_code}")
                
                # Add small delay to simulate realistic usage
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  {i+1:2d}. Failed: {str(e)}")
        
        print(f"\n✅ Completed {success_count}/{num_requests} successful requests")
    
    def show_analytics_summary(self):
        """Display analytics summary from the database"""
        print("\n📊 Analytics Summary")
        print("=" * 50)
        
        try:
            # Get analytics data from API
            response = requests.get(f"{self.api_url}/analytics?days=7", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                summary = data.get('summary', {})
                print(f"📈 Total Predictions: {summary.get('total_predictions', 0):,}")
                print(f"✅ Success Rate: {summary.get('success_rate', 0):.1f}%")
                print(f"⚡ Avg Inference Time: {summary.get('avg_inference_time', 0):.1f}ms")
                print(f"🎯 Avg Confidence: {summary.get('avg_confidence', 0):.1f}%")
                print(f"❌ Error Count: {summary.get('error_count', 0)}")
                
                # Show daily breakdown
                daily_data = data.get('daily_breakdown', [])
                if daily_data:
                    print(f"\n📅 Daily Activity (Last {len(daily_data)} days):")
                    for day in daily_data[:5]:  # Show last 5 days
                        print(f"  {day['date']}: {day['total']} predictions (Real: {day['real_count']}, Fake: {day['fake_count']})")
                
                # Show error patterns
                errors = data.get('error_patterns', [])
                if errors:
                    print(f"\n🐛 Error Patterns:")
                    for error in errors[:3]:  # Show top 3 errors
                        print(f"  {error['error']}: {error['count']} occurrences")
                
            else:
                print(f"❌ Failed to get analytics: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error getting analytics: {e}")
    
    def check_database_contents(self):
        """Check what's in the analytics database"""
        print("\n🗄️  Database Contents")
        print("=" * 50)
        
        if not Path(self.db_path).exists():
            print("❌ Analytics database not found")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check predictions table
            cursor.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = cursor.fetchone()[0]
            print(f"📊 Total predictions logged: {prediction_count}")
            
            if prediction_count > 0:
                # Get latest predictions
                cursor.execute("""
                    SELECT timestamp, prediction, confidence, inference_time_ms, success 
                    FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                recent = cursor.fetchall()
                print(f"\n🕒 Recent predictions:")
                for row in recent:
                    timestamp, prediction, confidence, inference_time, success = row
                    status = "✅" if success else "❌"
                    print(f"  {status} {timestamp}: {prediction} ({confidence:.1f}%) - {inference_time:.1f}ms")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error checking database: {e}")
    
    def show_dashboard_urls(self):
        """Show available dashboard URLs"""
        print("\n🌐 Enterprise Dashboard URLs")
        print("=" * 50)
        print(f"🏢 Main Dashboard: {self.api_url}/dashboard")
        print(f"📊 Analytics API: {self.api_url}/analytics")
        print(f"💾 Export Data: {self.api_url}/export/analytics")
        print(f"❤️  Health Check: {self.api_url}/health")
        print(f"ℹ️  Model Info: {self.api_url}/info")
        print(f"📖 API Docs: {self.api_url}/docs")
    
    def demonstrate_enterprise_features(self):
        """Full demonstration of enterprise features"""
        print("🏢 Enterprise Analytics Dashboard Demo")
        print("=" * 60)
        
        # Check server status
        if not self.check_server_status():
            return
        
        # Show available URLs
        self.show_dashboard_urls()
        
        # Check existing data
        self.check_database_contents()
        
        # Simulate some usage
        print(f"\n🎯 Would you like to simulate API usage to generate demo data?")
        response = input("Enter 'y' to simulate 10 API calls, or press Enter to skip: ").strip().lower()
        
        if response == 'y':
            self.simulate_api_usage(10)
            time.sleep(1)  # Brief pause
        
        # Show analytics
        self.show_analytics_summary()
        
        print(f"\n🎉 Demo Complete!")
        print(f"👉 Open {self.api_url}/dashboard in your browser to see the full enterprise dashboard")

def main():
    """Main demo function"""
    demo = DashboardDemo()
    demo.demonstrate_enterprise_features()

if __name__ == "__main__":
    main() 