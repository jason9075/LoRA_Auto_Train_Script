import os
import glob
import cv2
import argparse
import logging
import numpy as np
from blip_inference import blip_inference, init_blip
from tqdm import tqdm

logger = logging.getLogger()

BACKGROUND_COLOR = (255, 255, 255)
TARGET_SIZE = (512, 512)  # TODO resize problem
REMOVE_TEXT = True


def gen_style(source_dir, target_dir, trigger_word="my_art"):
    # create 1_face folder under target_dir
    face_dir = os.path.join(target_dir, "1_style")
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    index = 0
    ext_type = (".png", ".jpg", ".jpeg")
    model, transform, device = init_blip()
    for image_path in tqdm(glob.glob(os.path.join(source_dir, "*"))):
        if not image_path.lower().endswith(ext_type):
            continue

        image = cv2.imread(image_path)

        # resize if image is too large
        while True:
            h, w, _ = image.shape
            if max(h, w) < 1500:
                break
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        if REMOVE_TEXT:
            image = image[:-60, :-60, :]

        image = cv2.resize(image, TARGET_SIZE)
        # gen description
        description = blip_inference(model, transform, device, image_path)[0]

        # convert balck background to white
        if image_path.lower().endswith(".png"):
            image[np.where((image == [0, 0, 0]).all(axis=2))] = BACKGROUND_COLOR

        cv2.imwrite(os.path.join(face_dir, f"{index}.jpg"), image)
        path = os.path.join(face_dir, f"{index}.txt")
        with open(path, "w") as f:
            f.write(f"{trigger_word} style, {description}")

        index += 1

    logger.info(f"Gen {index} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args for model folder
    parser.add_argument("style", type=str, default="my_art")
    args = parser.parse_args()

    gen_style("origin_image", "train_image", trigger_word=args.style)
