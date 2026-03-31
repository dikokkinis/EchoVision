import os
import requests

def download_video(url: str, out_path: str):
    print(f"Downloading video from {url}...")

    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping download.")
        return
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Video saved successfully to {out_path}.")