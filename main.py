import os
import glob
import json
import zipfile
import timeit
import sys
import logging
import time
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from google.cloud import storage
from module.gen_example import gen_face_example
from module.gen_face import gen_face
from module.train_lora import train
from module.remove_down_blocks_weights import remove
from module.clean import clean_data
from google.api_core import retry
from dotenv import load_dotenv

load_dotenv()
# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join("tmp", "app.log"))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info("⭐Start app.")

# constant
project_id = "onyx-codex-383017"
subscription_id = "TrainJob-sub"
bucket_id = "lora-train-images"
key_file = "onyx-codex-383017-1cd126f48131.json"
sample_dir = os.path.join("output", "sample")
# Number of seconds the subscriber should listen for messages
WAIT_SEC = int(os.getenv("SUB_WAIT_SEC", 10))

# global
try:
    key_path = os.path.join(os.getcwd(), "key", key_file)
    creds = service_account.Credentials.from_service_account_file(
        os.path.join(key_path)
    )
except Exception as e:
    logger.error(f"Error: {e}")
    exit(1)

subscriber = pubsub_v1.SubscriberClient(credentials=creds)
subscription_path = subscriber.subscription_path(project_id, subscription_id)
client = storage.Client.from_service_account_json(key_path)
bucket = client.get_bucket(bucket_id)


def process_job(attr, msg_data):
    user_id = attr["user_id"]
    model_key = attr["model_key"]
    model_name = attr["model_name"]
    trigger_word = attr["trigger_word"]
    category = attr["category"]
    # deserialize json from msg_data
    meta_data = json.loads(msg_data, encoding="utf-8")

    # handle config
    config = handle_config_file(model_name, category)
    if config is None:
        logger.error(f"Not support category: {category}")
        return

    logger.info(f"🔹Download {user_id}/{model_key} images.zip .")
    download_image_from_gcp(model_key, user_id)

    logger.info("🔹Start data augmentation")
    if category == "face":
        gen_face("origin_image", "train_image", meta_data["gender"], trigger_word)

    logger.info("🔹Start training job.")
    train(config, meta_data["train"])

    logger.info("🔹Remove down blocks and weights.")
    if category == "face":
        remove(config["output_dir"])

    logger.info("🔹Generate sample image.")
    if category == "face":
        gen_face_example(
            config["output_dir"],
            meta_data,
            config["output_name"],
            sample_dir,
            trigger_word,
        )

    logger.info("🔹Upload sample images to gcp.")
    upload_sample_image_to_gcp(user_id, model_key)
    zip_and_upload_gcp(user_id, model_key, "output")
    zip_and_upload_gcp(user_id, model_key, "log")

    logger.info("🔹Clean train image, model and log.")
    clean_data()


def upload_sample_image_to_gcp(user_id, model_key):
    logger.info("Upload sample images to gcp.")
    for sample_path in glob.glob(os.path.join(sample_dir, "*.png")):
        sample_name = os.path.basename(sample_path)
        gcp_object = f"{user_id}/{model_key}/sample/{sample_name}"
        blob = bucket.blob(gcp_object)
        blob.upload_from_filename(sample_path)


def zip_and_upload_gcp(user_id, model_key, target_folder):
    gcp_object = f"{user_id}/{model_key}/{target_folder}.zip"
    local_file = os.path.join("tmp", f"{target_folder}.zip")

    # zip target folder
    with zipfile.ZipFile(local_file, "w") as zipObj:
        # Iterate over all the files in directory
        for folderName, _, filenames in os.walk(target_folder):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath)
    blob = bucket.blob(gcp_object)
    logger.info(f"Upload {local_file} to {gcp_object}.")
    blob.upload_from_filename(local_file)


def handle_config_file(model_name, category):
    if category == "face":
        with open(os.path.join("config", "face.json")) as f:
            config = json.load(f)
        # overwrite config
        config["train_data_dir"] = os.path.join(os.getcwd(), config["train_data_dir"])
        config["logging_dir"] = os.path.join(os.getcwd(), config["logging_dir"])
        config["output_dir"] = os.path.join(os.getcwd(), config["output_dir"])
        config["output_name"] = model_name
        config["train_batch_size"] = os.getenv("BATCH_SIZE", 1)

        return config

    logger.error(f"Not support category: {category}")
    raise ValueError(f"Not support category: {category}")


def message_handle(message):
    msg = str(message)
    logger.info(f"Received message: {json.dumps(msg)}.")

    # Acknowledges the received messages so they will not be sent again.
    subscriber.acknowledge(
        request={"subscription": subscription_path, "ack_ids": [message.ack_id]}
    )
    # process training job
    start_time = timeit.default_timer()
    try:
        process_job(message.message.attributes, message.message.data)
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
    elapsed = timeit.default_timer() - start_time
    # count minute and second
    logger.info(
        f"Process message in {int(elapsed) // 60} minutes {elapsed % 60:.2f} sec.)"
    )


def main():
    # because of long running jon (>10min), we need to close the connection after receive the message
    logger.info(f"Listening for messages on {subscription_path}..")
    with subscriber:
        while True:
            response = subscriber.pull(
                request={"subscription": subscription_path, "max_messages": 1},
                retry=retry.Retry(deadline=60),
            )

            if len(response.received_messages) == 0:
                logger.info(f"No message received. Wait {WAIT_SEC} sec.")
                time.sleep(WAIT_SEC)
                continue

            if len(response.received_messages) > 1:
                logger.error(
                    f"Receive more than one message. {response.received_messages}"
                )

            message = response.received_messages[0]
            message_handle(message)

            time.sleep(1)


def download_image_from_gcp(model_key, user_id):
    # download image from bucket
    gcp_object = f"{user_id}/{model_key}/origin_images/images.zip"
    blob = bucket.blob(gcp_object)
    zip_filename = os.path.join("tmp", "images.zip")
    blob.download_to_filename(zip_filename)

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall("origin_image")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt Exit.")
        sys.exit(0)
