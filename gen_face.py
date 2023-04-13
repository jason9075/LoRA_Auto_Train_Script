import numpy as np
import glob
import mediapipe as mp
import cv2

BACKGROUND_COLOR = (255, 255, 255)
FACE_NAME = "MY_NAME"
TARGET_SIZE = (512, 512)

# dilate kernel
KERNEL = np.ones((16, 16), np.uint8)


def main():
    mp_face_detection = mp.solutions.face_detection.FaceDetection()
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

    # Load image folder
    image_paths = glob.glob("origin_image/*.jpg")
    image_paths += glob.glob("origin_image/*.png")

    index = 0
    for image_path in image_paths:
        print(f"Processing {image_path}...")

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
            print(f"No face detected:{image_path}")
            continue

        if len(results.detections) > 1:
            print(f"More than one face detected:{image_path}")
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
            print(f"No landmark detected:{image_path}")
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
        """ # calculate face portion of width """
        """ h, w, _ = image.shape """
        """ face_width = xmax - xmin """
        """ portion = face_width / w """
        """ scale = 0.15 / portion """
        """ tiny_face = cv2.resize(image, (0, 0), fx=scale, fy=scale) """
        """ # gen same shape white image """
        """ white_image = np.ones(image.shape, dtype=np.uint8) * 255 """
        """ # paste tiny face to white image """
        """ start_x = int(0.4 * w) """
        """ start_y = int(0.15 * h) """
        """ white_image[ """
        """     start_y : start_y + tiny_face.shape[0], """
        """     start_x : start_x + tiny_face.shape[1], """
        """     :, """
        """ ] = tiny_face """

        image = cv2.resize(image, TARGET_SIZE)
        cv2.imwrite("train_image/1_face/{}.jpg".format(index), image)
        path = "train_image/1_face/{}.txt".format(index)
        with open(path, "w") as f:
            f.write(f"{FACE_NAME}, a person with {FACE_NAME} face, white background")

        # still blurry
        """ # gen far away face """
        """ image = cv2.resize(white_image, TARGET_SIZE) """
        """ cv2.imwrite("train_image/1_face/{}_far.jpg".format(index), image) """
        """ path = "train_image/1_face/{}_far.txt".format(index) """
        """ with open(path, "w") as f: """
        """     f.write(f"a far photo of {FACE_NAME} face, white background") """
        index += 1


if __name__ == "__main__":
    main()
