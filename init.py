import os
import subprocess
import sys

def setup_environment():
    if os.environ.get("VIRTUAL_ENV") is None:
        print("Error: You must run this script from within a virtual environment or run setup.bat")
        sys.exit(1)

    print("Activating virtual environment and installing dependencies...")
    
    result2 = subprocess.run(f"pip install -r requirements.txt", shell=True)

    print("INSTALL SUCCESSFUL PAKCAGES:")
    
    subprocess.run(f"pip list", shell=True)

    # if result.returncode == 0:
    #     print("Dependencies installed successfully.")
        
    #     print("Running test/run.py...")
    #     result_test = subprocess.run(f"python test/run.py", shell=True)
        
    #     if result_test.returncode == 0:
    #         print("Test script ran successfully.")
            
    #         print("Running server...")
    #         result_app = subprocess.run(f"python app.py", shell=True)
            
    #         if result_app.returncode == 0:
    #             print("App is running successfully.")
    #         else:
    #             print("Error running app.py.")
    #     else:
    #         print("Error running test/run.py.")
    # else:
    #     print("Error installing dependencies.")

if __name__ == "__main__":
    setup_environment()
