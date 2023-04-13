import cv2
import numpy as np
import mediapipe as mp

WRITE_ENABLE = True
NUM_OF_FRAMES_TO_SAVE = 60
VIDEO_FILE = "origin_video/jason2.mp4"
BACKGROUND_COLOR = (255, 255, 255)
FACE_PADDING = 0.2
PROMPT = "a person with ohwx face with white background"


def main():
    mp_face_detection = mp.solutions.face_detection.FaceDetection()
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

    # dilate kernel
    kernel = np.ones((8, 8), np.uint8)

    cap = cv2.VideoCapture(VIDEO_FILE)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frame_count:{frame_count}")
    grab_every_count = int(frame_count / NUM_OF_FRAMES_TO_SAVE)
    print(f"grab_every_count:{grab_every_count}")

    index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        results_detection = mp_face_detection.process(frame)

        # continue if no face
        if not results_detection.detections:
            print("no face")
            continue

        for detection in results_detection.detections:
            bbox = detection.location_data.relative_bounding_box
            xmin = int(bbox.xmin * frame_width)
            ymin = int(bbox.ymin * frame_height)
            xmax = int((bbox.xmin + bbox.width) * frame_width)
            ymax = int((bbox.ymin + bbox.height) * frame_height)

            results_mesh = mp_face_mesh.process(frame)
            if not results_mesh.multi_face_landmarks:
                print("no face mesh landmark")
                continue

            landmark_list = []
            face_landmarks = results_mesh.multi_face_landmarks[0]
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                landmark_list.append([x, y])

            landmark_array = np.array(landmark_list)
            landmark_convex = cv2.convexHull(np.array(landmark_list))

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [landmark_convex], 255)
            # erode
            mask = cv2.dilate(mask, kernel, iterations=3)
            frame[mask == 0] = BACKGROUND_COLOR

            xmin = landmark_array[:, 0].min() - int(frame_width * FACE_PADDING)
            xmax = landmark_array[:, 0].max() + int(frame_width * FACE_PADDING)
            ymin = landmark_array[:, 1].min() - int(frame_height * FACE_PADDING)
            ymax = landmark_array[:, 1].max() + int(frame_height * FACE_PADDING)

            # resize face width to 1/2 of the frame width
            face_width = xmax - xmin
            ratio = face_width / (frame_width / 2)

            face = frame[ymin:ymax, xmin:xmax, :]
            face = cv2.resize(
                face, (int(frame_width / ratio), int(frame_height / ratio))
            )
            h, w, _ = face.shape

            # paste face to middle of frame
            frame[:, :, :] = BACKGROUND_COLOR
            frame[
                frame_height // 2 - h // 2 : frame_height // 2 - h // 2 + h,
                frame_width // 2 - w // 2 : frame_width // 2 - w // 2 + w,
                :,
            ] = face

            break

        # crop middle
        if frame_height != frame_width:
            side_length = int(min(frame_height, frame_width) / 2)
            xcp = frame_width // 2
            ycp = frame_height // 2
            if frame_width < frame_height:
                crop_start = ycp - side_length
                crop_end = ycp + side_length
                frame = frame[crop_start:crop_end, :, :]
            else:
                crop_start = xcp - side_length
                crop_end = xcp + side_length
                frame = frame[:, crop_start:crop_end, :]
        frame = cv2.resize(frame, (512, 512))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Detection", frame)

        index += 1
        if WRITE_ENABLE and index % grab_every_count == 0:
            cv2.imwrite(f"train_image/1_face/image-{index}.jpg", frame)
            with open(f"train_image/1_face/image-{index}.txt", "w") as f:
                f.write(PROMPT)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
