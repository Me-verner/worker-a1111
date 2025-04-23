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

# Available models
MODEL_MAP = {
    "anime_v1": "AnimeV1.safetensors",
    "anime_v2": "AnimeV2.safetensors",
    "cinematic_v1": "CinematicV1.safetensors",
    "fantasy_v1": "FantasyV1.safetensors",
    "realistic_v1": "RealisticV1.safetensors",
    "photoreal_v1": "PhotorealV1.safetensors",
    "artistic_v1": "ArtisticV1.safetensors"
}

# Available LoRAs
LORA_MAP = {
    "detail_enhancer": "DetailEnhancer.safetensors",
    "style_booster": "StyleBooster.safetensors"
}

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
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_MAP.keys())}")
    
    model_path = f"/stable-diffusion-webui/models/Stable-diffusion/{MODEL_MAP[model_name]}"
    payload = {"sd_model_checkpoint": model_path}
    response = automatic_session.post(f"{LOCAL_URL}/options", json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to set model: {response.text}")

def run_inference(inference_request):
    response = automatic_session.post(f"{LOCAL_URL}/txt2img", json=inference_request, timeout=600)
    return response.json()

def handler(event):
    input_data = event["input"]
    
    prompt = input_data.get("prompt", "a photo of an astronaut riding a horse on mars")
    negative_prompt = input_data.get("negative_prompt", "blurry, bad quality")
    model_name = input_data.get("model_name", "realistic_v1")
    lora_name = input_data.get("lora_name", None)
    lora_weight = input_data.get("lora_weight", 0.7)
    width = input_data.get("width", 512)
    height = input_data.get("height", 512)
    num_inference_steps = input_data.get("num_inference_steps", 20)
    guidance_scale = input_data.get("guidance_scale", 7.5)
    seed = input_data.get("seed", 42)

    if lora_name and lora_name in LORA_MAP:
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
