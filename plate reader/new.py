import requests

# This is a direct RAW link to a proven YOLOv8 license plate detector
url = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"

print("Downloading the model... this is about 6MB.")
response = requests.get(url)

if response.status_code == 200:
    with open("best.pt", "wb") as f:
        f.write(response.content)
    print("✅ Success! 'best.pt' is now in your folder.")
else:
    print(f"❌ Failed to download. Error code: {response.status_code}")