import numpy as np
import os
import glob
import cv2
import argparse
import logging
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

logger = logging.getLogger()

BACKGROUND_COLOR = (255, 255, 255)
FACE_NAME = "MY_NAME"
TARGET_SIZE = (512, 512)

# dilate kernel
KERNEL = np.ones((16, 16), np.uint8)


def retry_face_detection_by_pose(mp_pose, mp_detect, image):
    h, w, _ = image.shape
    results = mp_pose.process(image)

    if results.pose_landmarks is None:
        return None, [0, 0, 1, 1]

    # find face area
    # https://learnopencv.com/wp-content/uploads/2022/03/MediaPipe-pose-BlazePose-Topology.jpg
    face_landmarks = results.pose_landmarks.landmark[0:10]
    face_landmarks = np.array(
        [
            [landmark.x * w, landmark.y * h, landmark.z * w]
            for landmark in face_landmarks
        ]
    )
    face_landmarks = face_landmarks.astype(np.int32)

    # find face bounding bbox
    xmin = np.min(face_landmarks[:, 0])
    ymin = np.min(face_landmarks[:, 1])
    xmax = np.max(face_landmarks[:, 0])
    ymax = np.max(face_landmarks[:, 1])

    # crop face area
    face_width = xmax - xmin
    crop_size = int(face_width * 2.0)
    xmin_crop = max(xmin - int(crop_size / 2), 0)
    xmax_crop = min(xmax + int(crop_size / 2), w)
    ymin_crop = max(ymin - int(crop_size / 2), 0)
    ymax_crop = min(ymax + int(crop_size / 2), h)
    image = image[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]

    results = mp_detect.process(image)

    factor = [
        xmin_crop / w,
        ymin_crop / h,
        (xmax_crop - xmin_crop) / w,
        (ymax_crop - ymin_crop) / h,
    ]
    return results, factor


def gen_face(source_dir, target_dir, gender, trigger_word=FACE_NAME):
    import mediapipe as mp

    mp_face_detection = mp.solutions.face_detection.FaceDetection()
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
    mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    # create 1_face folder under target_dir
    face_dir = os.path.join(target_dir, "1_face")
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    # convert .heic to .jpg
    for image_path in glob.glob(os.path.join(source_dir, "*")):
        if image_path.lower().endswith(".heic"):
            image = Image.open(image_path)
            image = image.convert("RGB")
            filename = os.path.basename(image_path).split(".")[0] + ".jpg"
            image.save(os.path.join(source_dir, filename))

    index = 0
    ext_type = (".png", ".jpg", ".jpeg")
    no_face_files = []
    more_than_one_face_files = []
    for image_path in glob.glob(os.path.join(source_dir, "*")):
        if not image_path.lower().endswith(ext_type):
            continue

        # find face
        image = cv2.imread(image_path)

        # resize if image width to 800 is too large
        h, w, _ = image.shape
        if w > 800:
            rough_image = cv2.resize(image, (800, int(h * 800 / w)))
        else:
            rough_image = image

        # Convert the BGR image to RGB before processing.
        rough_image = cv2.cvtColor(rough_image, cv2.COLOR_BGR2RGB)

        results = mp_face_detection.process(rough_image)

        addition_x = 0
        addition_y = 0
        scale_x = 1
        scale_y = 1

        # try again if no face is detected
        if results.detections is None:
            results, factor = retry_face_detection_by_pose(
                mp_pose, mp_face_detection, rough_image
            )
            addition_x = factor[0]
            addition_y = factor[1]
            scale_x = factor[2]
            scale_y = factor[3]

        # if still no face, skip this image
        if results is None:
            no_face_files.append(image_path)
            continue

        if len(results.detections) > 1:
            more_than_one_face_files.append(image_path)
            continue

        del rough_image

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        xmin = int((bbox.xmin * scale_x + addition_x) * w)
        ymin = int((bbox.ymin * scale_y + addition_y) * h)
        xmax = int((bbox.xmin * scale_x + addition_x + bbox.width * scale_x) * w)
        ymax = int((bbox.ymin * scale_y + addition_y + bbox.height * scale_y) * h)

        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        face_width = xmax - xmin

        # crop square of face to make it easier to detect landmarks
        crop_size = int(face_width * 1.2)
        xmin_crop = max(cx - crop_size, 0)
        xmax_crop = min(cx + crop_size, w)
        ymin_crop = max(cy - crop_size, 0)
        ymax_crop = min(cy + crop_size, h)
        image = image[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]
        # make it square if h, w are not equal
        h, w, _ = image.shape
        if h > w:
            image = cv2.copyMakeBorder(
                image,
                0,
                0,
                int((h - w) / 2),
                int((h - w) / 2),
                cv2.BORDER_CONSTANT,
                value=BACKGROUND_COLOR,
            )
        elif h < w:
            image = cv2.copyMakeBorder(
                image,
                int((w - h) / 2),
                int((w - h) / 2),
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=BACKGROUND_COLOR,
            )

        image = cv2.resize(image, TARGET_SIZE)
        h, w, _ = image.shape

        # draw face bounding box
        """ cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) """

        ######
        # mask out of face area
        landmark_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_mesh = mp_face_mesh.process(landmark_image)
        if results_mesh.multi_face_landmarks is None:
            no_face_files.append(image_path)
            continue

        for face_landmarks in results_mesh.multi_face_landmarks:
            landmark_list = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
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
    print(
        f"Gen {index} faces, no face: {no_face_files}, more than one face: {more_than_one_face_files}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args for model folder
    parser.add_argument("gender", type=str, default="male")
    args = parser.parse_args()

    gen_face("origin_image", "train_image", args.gender)
