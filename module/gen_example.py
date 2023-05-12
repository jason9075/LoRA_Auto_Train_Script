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
API_GET_SD_MODELS = "/sdapi/v1/sd-models"
API_POST_OPTIONS = "/sdapi/v1/options"
API_POST_TXT2IMG = "/sdapi/v1/txt2img"


def gen_face_example(lora_dir, meta_data, target_lora, img_output_dir, trigger_word):
    enable = meta_data["gen_example_enable"]
    if not enable:
        logger.info("gen_example is disabled.")
        return
    gender = meta_data["gender"]

    # check url is local host or not 127.0.0.1
    is_local_host = URL.find("127.0.0.1") != -1
    if is_local_host:
        # cp lora file to webui server
        subprocess.Popen(
            [
                "bash",
                "-c",
                f"cp {lora_dir}/{target_lora}.safetensors {SD_WEB_PATH}models/Lora/",
            ]
        )

        # 啟用shell script，例如啟用名為start_server.sh的腳本
        start_server = (
            #        f"./webui.sh --xformers --api --api-log --nowebui --lora-dir {lora_dir}"
            f"./webui.sh --xformers --api --api-log --nowebui"
        )
        p = subprocess.Popen(
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
            response = requests.get(f"{URL}{API_GET_SD_MODELS}")
            available_ckpt = response.json()
            available_ckpt = [ckpt["title"] for ckpt in available_ckpt]
            logging.info(
                f"Response: {response.status_code} SD WebUI is ready. Available checkpoints: {available_ckpt}"
            )

        except requests.exceptions.ConnectionError:
            if is_local_host:
                if p.poll() is not None:
                    raise ValueError("SD WebUI server is not running.")
            logging.info(f"Retrying... in {RETRY_SEC} seconds")
            time.sleep(RETRY_SEC)

    clothes = "shirt" if gender == "male" else "blouse"
    # payload
    payload = {
        "prompt": f"RAW photo, {gender}, a close portrait of {trigger_word}, wearing {clothes}, random background, high detailed skin, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, <lora:{target_lora}:1>",
        "negative_prompt": "(deformed iris, deformed pupils:1.3), naked, nude, nsfw, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        "sampler_name": "UniPC",
        "sampler_index": "UniPC",
        "steps": 20,
        "restore_faces": "false",
        "n_iter": 10,
        "width": 384,
        "height": 384,
        "seed": 9075,
    }
    # add example_dict if exists
    if meta_data.get("example") is not None:
        payload.update(meta_data.get("example"))

    # loop through all checkpoints
    logger.info(f"Generating images for {CKPT_LIST}...")
    for ckpt in tqdm(CKPT_LIST):
        ckpt_name = ckpt.split(".")[0]

        option_payload = {"sd_model_checkpoint": ckpt, "CLIP_stop_at_last_layers": 1}
        # update checkpoint file
        response = requests.post(url=f"{URL}{API_POST_OPTIONS}", json=option_payload)

        response = requests.post(url=f"{URL}{API_POST_TXT2IMG}", json=payload)

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

    if is_local_host:
        # shutdown server
        p.kill()

        # remove lora file from webui server
        subprocess.run(
            [
                "bash",
                "-c",
                f"rm {SD_WEB_PATH}/models/Lora/{target_lora}.safetensors",
            ]
        )


if __name__ == "__main__":
    lora_dir = os.path.join(os.getcwd(), "output", "model")
    output_dir = os.path.join(os.getcwd(), "output", "sample")
    meta_data = {"gender": "male", "gen_example_enable": True}
    gen_face_example(lora_dir, meta_data, "jason-model-cosine", output_dir, "MY_NAME")
