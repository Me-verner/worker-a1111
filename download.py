import os
import requests

def download_file(url, dest_folder, filename=None):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    if not filename:
        filename = url.split("/")[-1].split("?")[0]
    file_path = os.path.join(dest_folder, filename)
    print(f"🔽 Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"❌ Failed to download {url}")
        exit(1)

# Download models
print("🔽 Downloading models...")
with open('models.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        type_url = line.split(' ', 1)
        if len(type_url) != 2:
            print(f"❌ Invalid model entry: {line}")
            exit(1)
        model_type, url = type_url
        if model_type == "checkpoint":
            download_file(url, "downloads/checkpoints")
        elif model_type == "lora":
            download_file(url, "downloads/loras")
        else:
            print(f"❌ Unknown model type: {model_type}")
            exit(1)

# Download extensions
print("🔽 Downloading extensions...")
with open('extensions.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        url = line
        repo_name = url.rstrip('/').split('/')[-1]
        dest_dir = os.path.join("downloads", "extensions", repo_name)
        if not os.path.exists(dest_dir):
            os.system(f"git clone {url} {dest_dir}")
        else:
            print(f"✅ Extension {repo_name} already cloned.")
