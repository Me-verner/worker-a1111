import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import os
import json
import subprocess
import shutil
import time as time_module
import re

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
EXTENSIONS_DIR = "/stable-diffusion-webui/extensions"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FILE = os.path.join(SCRIPT_DIR, "models.txt")
EXTENSIONS_FILE = os.path.join(SCRIPT_DIR, "extensions.txt")

# Configure session with retries
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# Directory mappings for model types
directories = {
    "checkpoints": ("/stable-diffusion-webui/models/Stable-diffusion", [".safetensors", ".ckpt"]),
    "loras": ("/stable-diffusion-webui/models/Lora", [".safetensors", ".pt"]),
    "vaes": ("/stable-diffusion-webui/models/VAE", [".safetensors", ".pt"]),
    "embeddings": ("/stable-diffusion-webui/embeddings", [".pt", ".bin", ".safetensors"]),
}

# Map singular to plural model types
model_type_mapping = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "vae": "vaes",
    "embedding": "embeddings"
}

# Refresh endpoints for each model type
refresh_endpoints = {
    "checkpoints": "refresh-checkpoints",
    "loras": "refresh-loras",
    "vaes": "refresh-vae",
    "embeddings": "refresh-embeddings"
}

def format_size(size):
    """Convert file size to human-readable format."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"

def wait_for_service(url):
    """Wait for the WebUI service to be ready."""
    retries = 0
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)
        time.sleep(0.2)

def set_model(model_name):
    """Set the checkpoint model by its filename, stripping extension for title."""
    model_dir = directories["checkpoints"][0]
    model_path = os.path.join(model_dir, model_name)
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file {model_name} not found at {model_path}")

    model_title = os.path.splitext(model_name)[0]
    payload = {"sd_model_checkpoint": model_title}
    response = automatic_session.post(f"{LOCAL_URL}/options", json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to set model: {response.text}")

def refresh_model_type(model_type):
    """Refresh the model list for a specific type."""
    if model_type not in refresh_endpoints:
        raise ValueError(f"Invalid model type for refresh: {model_type}")
    endpoint = refresh_endpoints[model_type]
    response = automatic_session.post(f"{LOCAL_URL}/{endpoint}", timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to refresh {model_type}: {response.text}")
    print(f"{model_type.capitalize()} refreshed successfully.")

def restart_server():
    """Restart the WebUI server via API."""
    try:
        response = automatic_session.post(f"{LOCAL_URL}/server-restart", timeout=60)
        if response.status_code == 200:
            print("Server restart initiated.")
            return {"status": "restart initiated", "success": True}
        else:
            return {"status": f"Failed to restart server: HTTP {response.status_code} - {response.text}", "success": False}
    except Exception as e:
        return {"status": f"Failed to restart server: {str(e)}", "success": False}

def extract_filename(response):
    """Extract the filename from the response headers or URL."""
    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        filename_match = re.search(r'filename="?(.+?)"?(;|$)', content_disposition)
        if filename_match:
            return filename_match.group(1)
    url_filename = os.path.basename(response.url.split('?')[0])
    return url_filename if url_filename else "downloaded_model.safetensors"

def download_model(model_type, url, filename=None, token=None):
    """Download a model file from a URL and save it to the appropriate directory."""
    # Map singular to plural model type if necessary
    model_type = model_type_mapping.get(model_type, model_type)
    if model_type not in directories:
        raise ValueError(f"Invalid model type: {model_type}")

    dir_path, extensions = directories[model_type]

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code} {response.reason}")

    if not filename:
        filename = extract_filename(response)
        if not any(filename.endswith(ext) for ext in extensions):
            raise ValueError(f"Extracted filename {filename} does not have a valid extension for {model_type}")

    target_path = os.path.join(dir_path, filename)

    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Refresh the model list for this type
    refresh_model_type(model_type)

    size = os.path.getsize(target_path)
    modified = time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime(os.path.getmtime(target_path)))
    return {
        "name": filename,
        "type": model_type[:-1],
        "size": format_size(size),
        "modified": modified,
        "path": target_path
    }

def install_from_file(file_path, install_type):
    """Install models or extensions from a file."""
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if install_type == "models":
            try:
                model_type, url = line.split('|', 1)
                model_type = model_type.strip()
                url = url.strip()
                result = download_model(model_type, url)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "line": line})
        elif install_type == "extensions":
            try:
                extension_name = os.path.basename(line.rstrip('/'))
                extension_path = os.path.join(EXTENSIONS_DIR, extension_name)
                if os.path.exists(extension_path):
                    shutil.rmtree(extension_path)
                subprocess.run(["git", "clone", line, extension_path], check=True)
                results.append({"status": "installed", "extension": extension_name})
            except Exception as e:
                results.append({"error": str(e), "line": line})

    return results

def install_all():
    """Install all models and extensions from files."""
    models_results = install_from_file(MODELS_FILE, "models")
    extensions_results = install_from_file(EXTENSIONS_FILE, "extensions")

    if extensions_results:
        restart_result = restart_server()
        extensions_results.append(restart_result)

    return {
        "models": models_results,
        "extensions": extensions_results
    }

def install_models():
    """Install models from models.txt."""
    return install_from_file(MODELS_FILE, "models")

def install_extensions():
    """Install extensions from extensions.txt."""
    results = install_from_file(EXTENSIONS_FILE, "extensions")
    if results:
        restart_result = restart_server()
        results.append(restart_result)
    return results

def list_extensions():
    """List installed extensions."""
    if not os.path.exists(EXTENSIONS_DIR):
        return []

    extensions = []
    for item in os.listdir(EXTENSIONS_DIR):
        path = os.path.join(EXTENSIONS_DIR, item)
        if os.path.isdir(path):
            extensions.append(item)
    return extensions

def install_extension(url):
    """Install a single extension from a URL."""
    extension_name = os.path.basename(url.rstrip('/'))
    extension_path = os.path.join(EXTENSIONS_DIR, extension_name)
    if os.path.exists(extension_path):
        shutil.rmtree(extension_path)
    subprocess.run(["git", "clone", url, extension_path], check=True)
    restart_result = restart_server()
    return {"status": "installed", "extension": extension_name, "restart": restart_result}

def delete_extension(extension_name):
    """Delete an installed extension."""
    extension_path = os.path.join(EXTENSIONS_DIR, extension_name)
    if not os.path.exists(extension_path):
        raise ValueError(f"Extension {extension_name} not found")

    shutil.rmtree(extension_path)
    restart_result = restart_server()
    return {"status": "deleted", "extension": extension_name, "restart": restart_result}

def get_models():
    """List all available models with details."""
    result = {}
    for model_type, (dir_path, extensions) in directories.items():
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(tuple(extensions)) and os.path.isfile(os.path.join(dir_path, f))]
            file_list = []
            for file in files:
                path = os.path.join(dir_path, file)
                size = os.path.getsize(path)
                modified = time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime(os.path.getmtime(path)))
                file_list.append({
                    "name": file,
                    "type": model_type[:-1],
                    "size": format_size(size),
                    "modified": modified,
                    "path": path
                })
            result[model_type] = {
                "count": len(files),
                "files": file_list
            }
        else:
            result[model_type] = {
                "count": 0,
                "files": []
            }
    return result

def get_sd_models():
    """Get the list of available Stable Diffusion models from the API."""
    response = automatic_session.get(f"{LOCAL_URL}/sd-models", timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to get SD models: {response.text}")
    return response.json()

def get_options():
    """Get current WebUI options."""
    response = automatic_session.get(f"{LOCAL_URL}/options", timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to get options: {response.text}")
    return response.json()

def set_options(options):
    """Set WebUI options."""
    response = automatic_session.post(f"{LOCAL_URL}/options", json=options, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to set options: {response.text}")
    return {"status": "options updated"}

def get_progress():
    """Get current progress of an ongoing task."""
    response = automatic_session.get(f"{LOCAL_URL}/progress", timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to get progress: {response.text}")
    return response.json()

def inference_handler(input_data):
    """Handle image generation with flexible parameters."""
    model_name = input_data.get("model_name")
    if model_name:
        set_model(model_name)

    override_settings = input_data.get("override_settings", {})

    inference_request = {
        "prompt": input_data.get("prompt", ""),
        "negative_prompt": input_data.get("negative_prompt", ""),
        "steps": input_data.get("steps", 20),
        "width": input_data.get("width", 512),
        "height": input_data.get("height", 512),
        "cfg_scale": input_data.get("cfg_scale", 7.5),
        "seed": input_data.get("seed", -1),
        "override_settings": override_settings,
    }

    if input_data.get("enable_hr", False):
        inference_request["enable_hr"] = True
        inference_request["hr_scale"] = input_data.get("hr_scale", 2.0)
        inference_request["hr_upscaler"] = input_data.get("hr_upscaler", "Latent")
        inference_request["hr_second_pass_steps"] = input_data.get("hr_second_pass_steps", 20)
        inference_request["denoising_strength"] = input_data.get("denoising_strength", 0.55)

    response = automatic_session.post(f"{LOCAL_URL}/txt2img", json=inference_request, timeout=600)
    if response.status_code != 200:
        raise Exception(f"Failed to run inference: {response.text}")
    return response.json()

def img2img_handler(input_data):
    """Handle image-to-image generation."""
    model_name = input_data.get("model_name")
    if model_name:
        set_model(model_name)

    inference_request = {
        "init_images": input_data.get("init_images", []),
        "prompt": input_data.get("prompt", ""),
        "negative_prompt": input_data.get("negative_prompt", ""),
        "steps": input_data.get("steps", 20),
        "width": input_data.get("width", 512),
        "height": input_data.get("height", 512),
        "cfg_scale": input_data.get("cfg_scale", 7.5),
        "seed": input_data.get("seed", -1),
        "denoising_strength": input_data.get("denoising_strength", 0.75),
    }

    response = automatic_session.post(f"{LOCAL_URL}/img2img", json=inference_request, timeout=600)
    if response.status_code != 200:
        raise Exception(f"Failed to run img2img: {response.text}")
    return response.json()

def handler(event):
    """Main handler function to route actions."""
    input_data = event["input"]
    action = input_data.get("action", "inference")

    if action == "inference":
        return inference_handler(input_data)
    elif action == "img2img":
        return img2img_handler(input_data)
    elif action == "get_models":
        return get_models()
    elif action == "get_sd_models":
        return get_sd_models()
    elif action == "get_options":
        return get_options()
    elif action == "set_options":
        return set_options(input_data.get("options", {}))
    elif action == "get_progress":
        return get_progress()
    elif action == "install_all":
        return install_all()
    elif action == "install_models":
        return install_models()
    elif action == "install_extensions":
        return install_extensions()
    elif action == "list_extensions":
        return list_extensions()
    elif action == "install_extension":
        url = input_data.get("url")
        if not url:
            raise ValueError("URL is required for install_extension")
        return install_extension(url)
    elif action == "delete_extension":
        extension_name = input_data.get("extension_name")
        if not extension_name:
            raise ValueError("extension_name is required for delete_extension")
        return delete_extension(extension_name)
    elif action == "restart_server":
        return restart_server()
    elif action == "refresh_models":
        model_type = input_data.get("type")
        if model_type:
            refresh_model_type(model_type)
            return {"status": f"{model_type} refreshed"}
        else:
            for model_type in refresh_endpoints.keys():
                refresh_model_type(model_type)
            return {"status": "all models refreshed"}
    else:
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    wait_for_service(url=f"{LOCAL_URL}/sd-models")
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
