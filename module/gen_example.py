import json
import requests
import io
import base64
from tqdm import tqdm
from PIL import Image, PngImagePlugin
import logging
import time
import requests
import subprocess
import os

from dotenv import load_dotenv

logger = logging.getLogger()
load_dotenv()

SD_WEB_PATH = os.getenv("SD_WEB_PATH")

URL = "http://127.0.0.1:7861"
GENDER = "male"
STEPS = 50
SAMPLE_METHOD = "UniPC"
CKPT_LIST = [
    "v1-5-pruned-emaonly.safetensors [6ce0161689]",
    "chilloutmix_NiPrunedFp32Fix.safetensors [fc2511737a]",
    "deliberate_v2.safetensors [9aba26abdf]",
    "realisticVisionV20_v20.safetensors [c0d1994c73]",
]
RETRY_SEC = 20
BATCH_COUNT = 6
RESOLUTION = 384


# API List
GET_SD_MODELS = "/sdapi/v1/sd-models"
POST_OPTIONS = "/sdapi/v1/options"
POST_TXT2IMG = "/sdapi/v1/txt2img"


def gen_example(lora_dir, target_lora, img_output_dir, trigger_word):
    # 啟用shell script，例如啟用名為start_server.sh的腳本
    start_server = (
        f"./webui.sh --xformers --api --api-log --nowebui --lora-dir {lora_dir}"
    )
    subprocess.Popen(
        [
            "bash",
            "-c",
            f"cd {SD_WEB_PATH} && HF_HUB_DISABLE_TELEMETRY=1 && {start_server}",
        ]
    )

    # 發送HTTP請求直到回應為200
    response = None
    while response is None or response.status_code != 200:
        try:
            time.sleep(RETRY_SEC)
            response = requests.get(f"{URL}{GET_SD_MODELS}")
            logging.info(f"Response: {response.status_code} SD WebUI is ready.")
            available_ckpt = response.json()
            available_ckpt = [ckpt["title"] for ckpt in available_ckpt]
            # Limit to max
            logging.info(f"Available checkpoints: {available_ckpt}")

        except requests.exceptions.ConnectionError:
            logging.info(f"Retrying... in {RETRY_SEC} seconds")

    # payload
    payload = {
        "prompt": f"RAW photo, {GENDER}, a close photo of {trigger_word}, upper body, random background, high detailed skin, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3<lora:{target_lora}:1>",
        "negative_prompt": "(deformed iris, deformed pupils:1.3), nude, nsfw, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        "steps": STEPS,
        "sampler_name": "UniPC",
        "sampler_index": SAMPLE_METHOD,
        "restore_faces": "true",
        "n_iter": BATCH_COUNT,
        "width": RESOLUTION,
        "height": RESOLUTION,
        "seed": 9075,
    }

    # loop through all checkpoints
    logger.info(f"Generating images for {CKPT_LIST}...")
    for ckpt in tqdm(CKPT_LIST):
        ckpt_name = ckpt.split(".")[0]

        option_payload = {"sd_model_checkpoint": ckpt, "CLIP_stop_at_last_layers": 1}
        # update checkpoint file
        response = requests.post(url=f"{URL}{POST_OPTIONS}", json=option_payload)

        response = requests.post(url=f"{URL}{POST_TXT2IMG}", json=payload)

        r = response.json()

        for idx, i in enumerate(r["images"]):
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

            png_payload = {"image": "data:image/png;base64," + i}
            response2 = requests.post(url=f"{URL}/sdapi/v1/png-info", json=png_payload)

            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", response2.json().get("info"))
            image.save(
                os.path.join(img_output_dir, f"{ckpt_name}-{idx}.png"), pnginfo=pnginfo
            )

    # shutdown server
    subprocess.Popen(["pkill", "-f", "python3 launch.py"])
    subprocess.Popen(["pkill", "-f", "api_start.sh"])


if __name__ == "__main__":
    lora_dir = os.path.join(os.getcwd(), "output", "model")
    output_dir = os.path.join(os.getcwd(), "output", "sample")
    gen_example(lora_dir, "AutoTrainFace", output_dir, "MY_NAME")
