import logging
import glob
import os
import shutil

logger = logging.getLogger()


def clean_data():
    logger.info("Clean data.")
    # clean train_image only dir
    for filepath in glob.glob(os.path.join("train_image", "*")):
        if os.path.isdir(filepath):
            shutil.rmtree(
                filepath,
            )

    # clean origin image folder jpg and png
    ext_type = (".png", ".jpg", ".jpeg")
    for filepath in glob.glob(os.path.join("origin_image", "*")):
        if filepath.lower().endswith(ext_type):
            os.remove(filepath)

    # clean output folder
    for filepath in glob.glob(os.path.join("output", "model", "*.safetensors")):
        os.remove(filepath)

    for filepath in glob.glob(os.path.join("output", "sample", "*.png")):
        os.remove(filepath)

    # clean log folder
    for logdir in glob.glob(os.path.join("log", "*")):
        if os.path.isdir(logdir):
            shutil.rmtree(logdir)


def main():
    clean_data()


if __name__ == "__main__":
    main()
