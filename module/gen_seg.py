import numpy as np
import glob
import mediapipe as mp
import cv2
import time

BACKGROUND_COLOR = (255, 255, 255)
PERSON_NAME = "MY_NAME"

# dilate kernel
KERNEL = np.ones((16, 16), np.uint8)


def main():
    mp_face_detection = mp.solutions.face_detection.FaceDetection()
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # Load image folder
    image_paths = glob.glob("origin_image/*.jpg")
    image_paths += glob.glob("origin_image/*.png")

    index = 0
    for image_path in image_paths:
        with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=0
        ) as selfie_segmentation:
            image = cv2.imread(image_path)
            results = selfie_segmentation.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[condition[:, :, 0]] = 255
            mask = cv2.erode(mask, KERNEL, iterations=3)
            mask = cv2.dilate(mask, KERNEL, iterations=4)
            image[mask == 0] = BACKGROUND_COLOR

        image = cv2.resize(image, (512, 512))

        # debug
        """ cv2.imshow("MediaPipe Selfie Segmentation", image) """
        """ cv2.waitKey(0) """
        """ time.sleep(0.5) """
        """ continue """

        cv2.imwrite("train_image/1_face/{}.jpg".format(index), image)
        path = "train_image/1_face/{}.txt".format(index)
        with open(path, "w") as f:
            f.write(f"perosn with {PERSON_NAME} face, white background")

        index += 1


if __name__ == "__main__":
    main()
