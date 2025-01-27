import os
import subprocess

def setup_environment():
    print("Checking if virtual environment exists...")
    
    if not os.path.exists("venv"):
        print("Virtual environment not found. Setting it up...")
        result = subprocess.run(["python", "-m", "venv", "venv"])

        if result.returncode == 0:
            print("Virtual environment created successfully.")
        else:
            print("Error creating virtual environment.")
            return 
    else:
        print("Virtual environment already exists.")
    
    print("Activating virtual environment and installing dependencies...")
    
    if os.name == "nt":  
        activate_script = ".\\venv\\Scripts\\activate &&"
        
    result = subprocess.run(f"{activate_script} pip install -r requirements.txt", shell=True)
    
    if result.returncode == 0:
        print("Dependencies installed successfully.")
        
        print("Running test/run.py...")
        result_test = subprocess.run(f"python test/run.py", shell=True)
        
        if result_test.returncode == 0:
            print("Test script ran successfully.")
            
            print("Running server...")
            result_app = subprocess.run(f"python app.py", shell=True)
            
            if result_app.returncode == 0:
                print("App is running successfully.")
            else:
                print("Error running app.py.")
        else:
            print("Error running test/run.py.")
    else:
        print("Error installing dependencies.")

if __name__ == "__main__":
    setup_environment()
