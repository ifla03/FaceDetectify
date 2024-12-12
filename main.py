import cv2
import imutils
import argparse
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import face_recognition

def detect_faces(frame, face_cascade):
    # Resize frame for faster processing
    frame = imutils.resize(frame, width=min(800, frame.shape[1]))

    # Convert frame to grayscale for the face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    person_count = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Faces : {person_count}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('Output', frame)

    return frame, person_count, faces

def detectByPathImage(path, output_path, face_cascade):
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to load image at path {path}")
        return 0

    image, num_faces, _ = detect_faces(image, face_cascade)
    if output_path is not None:
        cv2.imwrite(output_path, image)
        print(f"Output image saved at {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return num_faces

def humanDetector(args):
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    image_path = args["image"]
    video_path = args["video"]
    camera = str(args["camera"]).lower() == 'true'

    if camera:
        print('[INFO] Opening Web Cam.')
        num_faces = detectByCamera(args["output"], face_cascade)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        max_faces = detectByPathVideo(video_path, args["output"], face_cascade)
        print(f"Total Number of Faces Detected in a Frame: {max_faces}")
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        num_faces = detectByPathImage(image_path, args["output"], face_cascade)
        print(f"Total Faces Detected in Image: {num_faces}")

def detectByCamera(output_path, face_cascade):   
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print('Failed to capture frame from camera. Exiting...')
        return

    print('Detecting faces... Press q to exit.')
    prev_frame = None
    total_faces = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame, num_faces, faces = detect_faces(frame, face_cascade)
        total_faces += num_faces

        key = cv2.waitKey(5)
        if key == ord('q') or num_faces > 0:
            print(f"Total Faces Detected: {num_faces}")
            cv2.imshow('Captured Image', frame)
            if output_path is not None:
                if prev_frame is None or not is_similar(prev_frame, frame):
                    cv2.imwrite(output_path, frame)
                    print(f"Output image saved at {output_path}")
                    prev_frame = frame
            cv2.waitKey(5000)  # Keep the window open for 5 seconds
            break

    video.release()
    cv2.destroyAllWindows()
    return total_faces

def is_similar(frame1, frame2, threshold=0.9):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold

def detectByPathVideo(path, output_path, face_cascade):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path)

    frame_count = 0
    max_faces = 0
    frame_interval = 50  # Process every 50 frames
    print('Detecting faces...')
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame, num_faces, faces = detect_faces(frame, face_cascade)

            if num_faces > max_faces:
                max_faces = num_faces

            if output_path is not None:
                frame_filename = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Output image saved at {frame_filename}")

        frame_count += 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return max_faces

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to output directory for saving frames")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    args = argsParser()
    humanDetector(args)
