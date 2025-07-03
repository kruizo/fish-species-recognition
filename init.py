import os
import subprocess
import importlib.util
import sys

def package_installed(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def setup_environment():
    """Main setup function with enhanced messaging."""
    print("🚀 Starting environment setup...")
    
    if os.environ.get("VIRTUAL_ENV") is None:
        print("❌ Error: You must run this script from within a virtual environment or run serve.bat")
        sys.exit(1)
    
    print("✅ Running in virtual environment.")

    print("📋 Installing requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", "requirements.txt",
            "--no-deps"
        ])
        print("✅ Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

    if not package_installed("torch"):
        print("🔥 Installing PyTorch + CUDA dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("✅ PyTorch with CUDA installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyTorch: {e}")
            sys.exit(1)
    else:
        print("✅ PyTorch already installed.")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA is available! Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   📱 GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA is not available. Training will use CPU.")
    except ImportError:
        print("⚠️  Could not import torch to check CUDA availability.")

    print("📦 Listing installed packages:")
    try:
        subprocess.run([sys.executable, "-m", "pip", "list"], shell=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to list packages.")
    
    print("🎉 Environment setup completed successfully!")

if __name__ == "__main__":
    success = setup_environment()
    if not success:
        sys.exit(1)  
    sys.exit(0)