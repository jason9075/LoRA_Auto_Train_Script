import os
import glob
import json
import zipfile
import timeit
import shutil
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from google.cloud import storage
from module.gen_example import gen_example
from module.gen_face import gen_face
from module.train_lora import train
from module.remove_down_blocks_weights import remove
import logging

# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join("tmp", "app.log"))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Start app.")

# constant
project_id = "onyx-codex-383017"
subscription_id = "TrainJob-sub"
bucket_id = "lora-train-images"
key_file = "onyx-codex-383017-1cd126f48131.json"
# Number of seconds the subscriber should listen for messages
timeout = 10.0

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
client = storage.Client.from_service_account_json(key_path)
bucket = client.get_bucket(bucket_id)


def process_job(attr, msg_data):
    user_id = attr["user_id"]
    model_key = attr["model_key"]
    trigger_word = attr["trigger_word"]
    category = attr["category"]

    download_image_from_gcp(model_key, user_id)

    # handle attr
    config = handle_attr(user_id, model_key, trigger_word, category)
    if config is None:
        logger.error(f"Not support category: {category}")
        return

    # start traing job
    train(config)

    # remove down blocks and weights
    if category == "face":
        remove(config["output_dir"])

    # generate some sample image
    if category == "face":
        sample_dir = os.path.join("output", "sample")
        gen_example(
            config["output_dir"],
            config["output_name"],
            sample_dir,
            trigger_word,
        )
        # upload sample image to gcp
        upload_sample_image_to_gcp(user_id, model_key, sample_dir)

    # zip output and log folder to gcp
    zip_and_upload_gcp(user_id, model_key, "output")
    zip_and_upload_gcp(user_id, model_key, "log")

    # clean train image, model and log
    clean_data()


def upload_sample_image_to_gcp(user_id, model_key, sample_dir):
    logger.info("Upload sample images to gcp.")
    for sample in glob.glob(os.path.join(sample_dir, "*.png")):
        gcp_object = f"{user_id}/{model_key}/sample/{sample}"
        blob = bucket.blob(gcp_object)
        blob.upload_from_filename(sample)


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


def handle_attr(user_id, model_key, trigger_word, category):
    if category == "face":
        # use origin image to gen train_image
        gen_face("origin_image", "train_image", trigger_word)
        with open(os.path.join("config", "face.json")) as f:
            config = json.load(f)
        config["train_data_dir"] = os.path.join(os.getcwd(), config["train_data_dir"])
        config["logging_dir"] = os.path.join(os.getcwd(), config["logging_dir"])
        config["output_dir"] = os.path.join(os.getcwd(), config["output_dir"])
        config["train_batch_size"] = os.getenv("BATCH_SIZE", 1)

        return config

    return None


def main():
    # Conbine the subscription_path = 'projects/{project_id}/subscriptions/{subscription_id}'
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    flow_control = pubsub_v1.types.FlowControl(max_messages=1)

    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=message_callback, flow_control=flow_control
    )
    logger.info(f"Listening for messages on {subscription_path}..")

    # Wrap subscriber in a 'with' block to automatically call close() when done.
    with subscriber:
        try:
            # When `timeout` is not set, result() will block indefinitely,
            # unless an exception is encountered first.
            streaming_pull_future.result(timeout=timeout)
        except TimeoutError:
            streaming_pull_future.cancel()


def message_callback(message):
    logger.info(f"Received message: {message}")
    # process message
    start_time = timeit.default_timer()
    message.ack()
    process_job(message.attributes, message.data)
    elapsed = timeit.default_timer() - start_time
    # compute minute and second
    logger.info(f"Process message in {elapsed // 60} minutes {elapsed % 60} sec.)")


def download_image_from_gcp(model_key, user_id):
    # download image from bucket
    gcp_object = f"{user_id}/{model_key}/origin_images/images.zip"
    blob = bucket.blob(gcp_object)
    zip_filename = os.path.join("tmp", "images.zip")
    logger.info(f"Download {gcp_object} to {zip_filename}.")
    blob.download_to_filename(zip_filename)

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall("origin_image")


if __name__ == "__main__":
    main()
