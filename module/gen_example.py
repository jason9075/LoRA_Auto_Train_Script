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
CKPT_LIST = [
    "v1-5-pruned-emaonly.safetensors [6ce0161689]",
    "chilloutmix_NiPrunedFp32Fix.safetensors [fc2511737a]",
    "deliberate_v2.safetensors [9aba26abdf]",
    "realisticVisionV20_v20.safetensors [c0d1994c73]",
]
RETRY_SEC = 20


# API List
GET_SD_MODELS = "/sdapi/v1/sd-models"
POST_OPTIONS = "/sdapi/v1/options"
POST_TXT2IMG = "/sdapi/v1/txt2img"


def gen_example(lora_dir, meta_data, target_lora, img_output_dir, trigger_word):
    logger.info("Start gen_example")
    gender = meta_data["gender"]
    batch_count = meta_data["batch_count"]
    sample_res = meta_data["sample_res"]
    sample_method = meta_data["sample_method"]
    sample_steps = meta_data["sample_steps"]

    # cp lora file to webui server
    subprocess.Popen(
        [
            "bash",
            "-c",
            f"cp {lora_dir}/{target_lora}.safetensors {SD_WEB_PATH}/models/Lora/",
        ]
    )

    # 啟用shell script，例如啟用名為start_server.sh的腳本
    start_server = (
        #        f"./webui.sh --xformers --api --api-log --nowebui --lora-dir {lora_dir}"
        f"./webui.sh --xformers --api --api-log --nowebui"
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
            available_ckpt = response.json()
            available_ckpt = [ckpt["title"] for ckpt in available_ckpt]
            logging.info(
                f"Response: {response.status_code} SD WebUI is ready. Available checkpoints: {available_ckpt}"
            )

        except requests.exceptions.ConnectionError:
            logging.info(f"Retrying... in {RETRY_SEC} seconds")

    clothes = "shirt" if gender == "male" else "blouse"
    # payload
    payload = {
        "prompt": f"RAW photo, {gender}, a close portrait of {trigger_word}, wearing {clothes}, random background, high detailed skin, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3<lora:{target_lora}:1>",
        "negative_prompt": "(deformed iris, deformed pupils:1.3), naked,nude, nsfw, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        "steps": sample_steps,
        "sampler_name": sample_method,
        "sampler_index": sample_method,
        "restore_faces": "true",
        "n_iter": batch_count,
        "width": sample_res,
        "height": sample_res,
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

    # remove lora file from webui server
    subprocess.Popen(
        [
            "bash",
            "-c",
            f"rm {SD_WEB_PATH}/models/Lora/{target_lora}.safetensors",
        ]
    )


if __name__ == "__main__":
    lora_dir = os.path.join(os.getcwd(), "output", "model")
    output_dir = os.path.join(os.getcwd(), "output", "sample")
    meta_data = {
        "batch_count": 3,
        "sample_method": "UniPC",
        "sample_res": 384,
        "sample_steps": 50,
        "gender": "male",
    }
    gen_example(lora_dir, meta_data, "AutoTrainFace", output_dir, "MY_NAME")
