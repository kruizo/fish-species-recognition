# run_browser.py
import subprocess
import webbrowser
import time
import sys
import os
import requests
from urllib.parse import urlparse

def wait_for_server(url, timeout=30):
    """Wait for the server to be ready."""
    print(f"Waiting for server to start at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    
    print("Server didn't start within timeout period.")
    return False

def main():
    print("Starting Flask app...")
    flask_process = subprocess.Popen([sys.executable, "app.py"])

    try:
        port = os.getenv('PORT', '8000')
        url = f"http://localhost:{port}/"
        
        if wait_for_server(url):
            print(f"Opening browser at {url}")
            webbrowser.open(url)
        else:
            print("Server failed to start, opening browser anyway...")
            webbrowser.open(url)

        flask_process.wait()
    except KeyboardInterrupt:
        print("Interrupted, terminating Flask app...")
        flask_process.terminate()
        flask_process.wait()

if __name__ == "__main__":
    main()