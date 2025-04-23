import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import os
import json

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
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
    model_path = f"/stable-diffusion-webui/models/Stable-diffusion/{model_name}"
    if not os.path.exists(model_path):
        raise ValueError(f"Model file {model_name} not found in /stable-diffusion-webui/models/Stable-diffusion")
    
    payload = {"sd_model_checkpoint": model_path}
    response = automatic_session.post(f"{LOCAL_URL}/options", json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to set model: {response.text}")

def get_available_models():
    # Get list of model files (without .safetensors extension) and LoRAs
    model_dir = "/stable-diffusion-webui/models/Stable-diffusion"
    lora_dir = "/stable-diffusion-webui/models/Lora"
    
    models = []
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(".safetensors"):
                models.append(file)  # Keep full filename including .safetensors
    
    loras = []
    if os.path.exists(lora_dir):
        for file in os.listdir(lora_dir):
            if file.endswith(".safetensors"):
                loras.append(file)  # Keep full filename including .safetensors
    
    return {
        "models": models,
        "loras": loras
    }

def run_inference(inference_request):
    response = automatic_session.post(f"{LOCAL_URL}/txt2img", json=inference_request, timeout=600)
    return response.json()

def handler(event):
    input_data = event["input"]
    
    # Handle request for available models
    if input_data.get("action") == "get_models":
        return get_available_models()

    # Handle inference request
    prompt = input_data.get("prompt", "a photo of an astronaut riding a horse on mars")
    negative_prompt = input_data.get("negative_prompt", "blurry, bad quality")
    model_name = input_data.get("model_name", "RealisticV1.safetensors")
    lora_name = input_data.get("lora_name", None)
    lora_weight = input_data.get("lora_weight", 0.7)
    width = input_data.get("width", 512)
    height = input_data.get("height", 512)
    num_inference_steps = input_data.get("num_inference_steps", 20)
    guidance_scale = input_data.get("guidance_scale", 7.5)
    seed = input_data.get("seed", 42)

    if lora_name:
        prompt += f" <lora:{lora_name}:{lora_weight}>"

    set_model(model_name)

    inference_request = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": num_inference_steps,
        "cfg_scale": guidance_scale,
        "seed": seed
    }

    result = run_inference(inference_request)
    return result

if __name__ == "__main__":
    wait_for_service(url=f"{LOCAL_URL}/sd-models")
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
