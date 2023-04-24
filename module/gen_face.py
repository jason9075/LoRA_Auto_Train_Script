import numpy as np
import os
import glob
import mediapipe as mp
import cv2
from torch import conj
import argparse
import logging

logger = logging.getLogger()

BACKGROUND_COLOR = (255, 255, 255)
FACE_NAME = "MY_NAME"
TARGET_SIZE = (512, 512)

# dilate kernel
KERNEL = np.ones((16, 16), np.uint8)


mp_face_detection = mp.solutions.face_detection.FaceDetection()
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()


def gen_face(source_dir, target_dir, gender, trigger_word=FACE_NAME):
    # create 1_face folder under target_dir
    face_dir = os.path.join(target_dir, "1_face")
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    index = 0
    ext_type = (".png", ".jpg", ".jpeg")
    no_face_files = []
    more_than_one_face_files = []
    for image_path in glob.glob(os.path.join(source_dir, "*")):
        if not image_path.lower().endswith(ext_type):
            continue

        # find face
        image = cv2.imread(image_path)

        # resize if image is too large
        h, w, _ = image.shape
        if min(h, w) > 1500:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        # Convert the BGR image to RGB before processing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        results = mp_face_detection.process(image)

        if results.detections is None:
            no_face_files.append(image_path)
            continue

        if len(results.detections) > 1:
            more_than_one_face_files.append(image_path)
            continue

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        xmax = int((bbox.xmin + bbox.width) * w)
        ymax = int((bbox.ymin + bbox.height) * h)

        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        # crop square of face to make it easier to detect landmarks
        if h != w:
            side_length = int(min(h, w) / 2)
            if w < h:
                crop_start = cy - side_length
                crop_end = cy + side_length
                if crop_start < 0:
                    crop_end = crop_end - crop_start
                    crop_start = 0
                if h < crop_end:
                    crop_start = crop_start - (crop_end - h)
                    crop_end = h
                image = image[crop_start:crop_end, :, :]
                ymin = ymin - crop_start
                ymax = ymax - crop_start
            else:
                crop_start = cx - side_length
                crop_end = cx + side_length
                if crop_start < 0:
                    crop_end = crop_end - crop_start
                    crop_start = 0
                if w < crop_end:
                    crop_start = crop_start - (crop_end - h)
                    crop_end = h
                image = image[:, crop_start:crop_end, :]
                xmin = xmin - crop_start
                xmax = xmax - crop_start

        # margin face for better landmark detections
        margin = 0.2
        xmin = max(int(xmin - (xmax - xmin) * margin), 0)
        xmax = min(int(xmax + (xmax - xmin) * margin), w)
        ymin = max(int(ymin - (ymax - ymin) * margin), 0)
        ymax = min(int(ymax + (ymax - ymin) * margin), h)
        landmark_image = image[ymin:ymax, xmin:xmax, :]
        h, w, _ = landmark_image.shape

        # draw face bounding box
        """ cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) """

        ######
        # mask out of face area
        results_mesh = mp_face_mesh.process(landmark_image)
        if results_mesh.multi_face_landmarks is None:
            no_face_files.append(image_path)
            continue

        for face_landmarks in results_mesh.multi_face_landmarks:
            landmark_list = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w + xmin)
                y = int(landmark.y * h + ymin)
                landmark_list.append([x, y])

            # draw landmark_list
            """ for landmark in landmark_list: """
            """     cv2.circle(image, (landmark[0], landmark[1]), 1, (0, 0, 255), -1) """
            landmark_array = cv2.convexHull(np.array(landmark_list))

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [landmark_array], 255)
            # erode
            mask = cv2.dilate(mask, KERNEL, iterations=3)
            # draw mask
            image[mask == 0] = BACKGROUND_COLOR

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.resize(image, TARGET_SIZE)
        cv2.imwrite(os.path.join(face_dir, f"{index}.jpg"), image)
        path = os.path.join(face_dir, f"{index}.txt")
        with open(path, "w") as f:
            f.write(
                f"{trigger_word}, 1{gender}, a person with {trigger_word} face, white background"
            )

        index += 1

    logger.info(
        f"Gen {index} faces, no face: {no_face_files}, more than one face: {more_than_one_face_files}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args for model folder
    parser.add_argument("gender", type=str, default="male")
    args = parser.parse_args()

    gen_face("origin_image", "train_image", args.gender)
