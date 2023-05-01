from re import sub
import os
import glob
import cv2
import argparse
import logging
from blip_inference import blip_inference, init_blip
from tqdm import tqdm

logger = logging.getLogger()

TRIGGER_WORD = "sys"
TARGET_SIZE = (512, 512)


def gen_aug(source_dir, target_dir, trigger_word=TRIGGER_WORD):
    # create 1_character folder under target_dir
    subject_folder = os.path.join(target_dir, "1_character")
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    index = 0
    ext_type = (".png", ".jpg", ".jpeg")

    model, transform, device = init_blip()
    for image_path in tqdm(glob.glob(os.path.join(source_dir, "*"))):
        if not image_path.lower().endswith(ext_type):
            continue

        image = cv2.imread(image_path)

        # resize if image is too large
        h, w, _ = image.shape
        if min(h, w) > 1500:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        # gen description
        description = blip_inference(model, transform, device, image_path)[0]

        image = cv2.resize(image, TARGET_SIZE)
        cv2.imwrite(os.path.join(subject_folder, f"{index}.jpg"), image)
        path = os.path.join(subject_folder, f"{index}.txt")
        with open(path, "w") as f:
            f.write(f"character, {trigger_word}, {description}")

        index += 1

    logger.info(f"Gen {index} subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args for model folder
    parser.add_argument("trigger_word", type=str, default=TRIGGER_WORD)
    args = parser.parse_args()

    gen_aug("origin_image", "train_image", args.trigger_word)
