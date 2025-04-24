"""
Run script for HADR Sentiment Analysis application
"""
import os
import sys
import subprocess
from pathlib import Path

def run_streamlit():
    """Run the Streamlit application"""
    app_path = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501"]
    
    print("=" * 80)
    print("Starting HADR Sentiment Analysis application...")
    print(f"App path: {app_path}")
    print("=" * 80)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    run_streamlit()
