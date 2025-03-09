from operator import __not__
import os
import torch 
import tensorflow_hub as hub

def run_test_processor():
    print("Running test_processor...")

    print(tf.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print(f"Using device: {device}")

    print("Test complete.")

if __name__ == "__main__":
    run_test_processor()