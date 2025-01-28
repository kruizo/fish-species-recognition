import os
import subprocess
import sys

def setup_environment():
    if os.environ.get("VIRTUAL_ENV") is None:
        print("Error: You must run this script from within a virtual environment or run setup.bat")
        sys.exit(1)

    print("Activating virtual environment and installing dependencies...")
    
    result = subprocess.run(f"pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", shell=True)

    if(result.returncode != 0):
        print("Error installing torch, torchvision, torchaudio")
        sys.exit(1)

    result2 = subprocess.run(f"pip install -r requirements.txt", shell=True)

    if(result2.returncode != 0):
        print("Error installing dependencies")
        sys.exit(1)
    
    subprocess.run(f"pip list", shell=True)

if __name__ == "__main__":
    setup_environment()
