import os
import gdown

weights_path = "backend/models/weights/"

weights = {
    'FINAL_torch_weights_baselineb16notauglr001notenhancedtraintime_freezeAllEpoch30withclassweights.pth': "1CkfTZ81TSvR06fDlm0CUX4b6YqyH-lrc",
    'FINAL_torch_weights_proposedb16notauglr001ENHANCEDwUNETtraintime_freezeAllEpoch30withclassweights.pth': "1_rrJbveTWo8eBpeKy9Ln_3s3Qg66huuF",
    'densenet_full_model_30epoch.pth': "1Gd4z-Dtjzzqn0McXyBe4GjklxipVVXdv",
    'inception_model_30epoch.pth': "1pBfOCNZk8z8-hD2kn-Jjnih4z8Y6WvOt",
    'latestmobilenet_v2_model_30epoch.pth': "1Fde1-kFxYKITk4lDU77RRa9QBe-qXPrC",
    'FINAL_unet_model224.pth': "1ggSkLLB3u5LeIjBYfkYmVbp1R86hyFhS",
    'FINAL_RRDB_ESRGAN_x4.pth': "1M8GQgbFdxT0XF4ZAE_RIjcicfTq_CHvo"
}

os.makedirs(weights_path, exist_ok=True)
print(f"✅ Checking the availability of models...")

for filename, file_id in weights.items():
    file_path = os.path.join(weights_path, filename)
    
    if not os.path.exists(file_path):  
        print(f"❌ {filename} 🔽 Downloading...")
        try:
            result = gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
            if not result:
                print(f"❌ Failed to download {filename} (Check Google Drive ID)")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    else:
        print(f"✅ {filename} already exists, skipping download.")

