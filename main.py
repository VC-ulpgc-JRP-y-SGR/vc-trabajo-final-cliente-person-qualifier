import cv2
import logging
import queue
import threading
import argparse
from face_detector.mediapipe import MediaPipeFaceDetector
from server import CameraServer

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ip", help="IP address of the server", default="127.0.0.1")
parser.add_argument("--port", help="Port of the camera", default="500")
args = parser.parse_args()

# Initialize camera and server
cap = cv2.VideoCapture(0)
server = CameraServer(ip=args.ip, buffer_size=4096, port=args.port)
server.start()

# Initialize queue and flag for thread communication
frame_queue = queue.Queue()
stop_flag = threading.Event()

# Function for the sending thread
def send_frames():
    while not stop_flag.is_set():
        while not frame_queue.empty():
            # Check if the queue has more than 10 frames
            if frame_queue.qsize() > 10:
                # Discard the oldest frame
                frame_queue.get_nowait()
            else:
                break

        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                server.send_frame(frame)
            except Exception as e:
                logging.error(f"Failed to send frame: {e}")
                stop_flag.set()


# Start the sending thread
send_thread = threading.Thread(target=send_frames)
send_thread.start()

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        frame_queue.put(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    logging.error(f"Error in main loop: {e}")
finally:
    # Signal the sending thread to stop and wait for it
    stop_flag.set()
    send_thread.join()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    server.close()
