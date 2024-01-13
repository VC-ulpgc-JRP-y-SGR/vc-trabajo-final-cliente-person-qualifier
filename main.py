import cv2
import logging
from face_detector.mediapipe import MediaPipeFaceDetector
from face_detector.viola_jones import ViolaJonesFaceDetector
from server import CameraServer

cap = cv2.VideoCapture(0)
server = CameraServer(ip="127.0.0.1", buffer_size= 4096, port=5005)
server.start()

while True:
    # Read frame from camera
    # Send frame
    try:
        frame = cap.read()[1]
        face_detector = MediaPipeFaceDetector()
        for face in face_detector.detect(frame):
            cv2.rectangle(frame, (face.bounding_box.origin.x, face.bounding_box.origin.y), (face.bounding_box.end.x, face.bounding_box.end.y), (255, 0, 0), 2)
        server.send_frame(frame)
    except Exception as e:
        logging.error(f"Failed to send frame: {e}")
        break

# Close connection
server.close()