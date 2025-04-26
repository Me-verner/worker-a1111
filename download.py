import os
import requests

def download_file(url, target_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def process_filelist(filelist_path, base_dir):
    if not os.path.exists(filelist_path):
        print(f"Warning: {filelist_path} does not exist. Skipping...")
        return

    with open(filelist_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"No entries in {filelist_path}. Skipping...")
        return

    os.makedirs(base_dir, exist_ok=True)

    for line in lines:
        if "|" not in line:
            print(f"Invalid line in {filelist_path}: {line}")
            continue

        type_part, url_part = line.split("|", 1)
        filename = os.path.basename(url_part.split("?")[0])
        save_path = os.path.join(base_dir, filename)

        if os.path.exists(save_path):
            print(f"Already exists: {save_path}")
            continue

        print(f"Downloading {filename} to {save_path}")
        download_file(url_part, save_path)

# Models download
process_filelist("/models.txt", "/stable-diffusion-webui/models/Stable-diffusion")

# Extensions install
process_filelist("/extensions.txt", "/stable-diffusion-webui/extensions")
