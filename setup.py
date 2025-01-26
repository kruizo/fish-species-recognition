import os
import subprocess

def setup_environment():
    print("Setting up virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"])
    print("Activating virtual environment and installing dependencies...")
    if os.name == "nt":  
        activate_script = ".\\venv\\Scripts\\activate &&"
    else:  
        activate_script = "source ./venv/bin/activate &&"
    
    subprocess.run(f"{activate_script} pip install -r requirements.txt", shell=True)
    print("Environment setup complete.")

if __name__ == "__main__":
    setup_environment()
