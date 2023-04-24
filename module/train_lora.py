import json
import timeit
import os
import glob
import subprocess
import logging
from dotenv import load_dotenv

logger = logging.getLogger()
load_dotenv()

TRAIN_PATH = os.getenv("SD_TRAIN_PATH")


def train(config, train_dict=None):
    # overwrite config
    if train_dict:
        for key, value in train_dict.items():
            config[key] = value
    max_train_steps = int(config["max_train_steps"])
    train_batch_size = int(config["train_batch_size"])
    train_data_dir = config["train_data_dir"]
    num_ckpts = 5

    # count num of train images
    image_count = len(glob.glob(os.path.join(train_data_dir, "**", "*.jpg")))
    if image_count == 0:
        logger.error("No images found in train_data_dir")
        raise ValueError("No images found in train_data_dir")

    max_train_steps = max_train_steps // train_batch_size
    image_per_steps = image_count // train_batch_size
    save_every_n_epochs = int(max_train_steps / image_per_steps / num_ckpts)

    # overwrite config
    config["max_train_steps"] = max_train_steps
    config["save_every_n_epochs"] = save_every_n_epochs

    logger.info(
        f"image_count: {image_count}, save_every_n_epochs: {save_every_n_epochs}"
    )
    train_network_args = []
    for key, value in config.items():
        if value == "":
            continue
        if isinstance(value, bool):
            if value:
                train_network_args.append(f"--{key}")
            continue
        train_network_args.append(f"--{key}")
        train_network_args.append(str(value))

    accelerate_args = [
        "accelerate",
        "launch",
        "--num_processes=1",
        "--num_cpu_threads_per_process=2",
        "--num_machines=1",
        "--mixed_precision=no",
        "train_network.py",
    ]

    logger.info("Execute command: " + " ".join(accelerate_args + train_network_args))
    start = timeit.default_timer()
    p = subprocess.run(
        [
            "bash",
            "-c",
            f"cd {TRAIN_PATH} && {' '.join(accelerate_args + train_network_args)}",
        ]
    )
    if p.returncode != 0:
        logger.error("Training failed.")
        raise ValueError("Training failed.")
    end = timeit.default_timer()
    logger.info(f"Training time: {end - start} sec.")


if __name__ == "__main__":
    with open("config/face.json") as f:
        config = json.load(f)
    train(config)
