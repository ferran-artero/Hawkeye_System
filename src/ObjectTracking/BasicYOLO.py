from ultralytics import YOLO
import cv2
import os

# Cambiar directorio actual a la carpeta donde está el script (o YAML)
os.chdir(r"c:/Users/ferra/OneDrive/Desktop/thws/Robotics Project - Hawkeye System/Hawkeye_System/src/ObjectTracking")

# Load YOLOv8 model (nano is fast and light)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt for better accuracy

# Open webcam or video file
cam = cv2.VideoCapture(1)  # Use 0 for webcam, or replace with 'video.mp4'

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Run tracking on the frame (class 0 = person) (, classes=[0])
    results = model.track(frame, persist=True, tracker="mybytetrack.yaml", conf=0.5, classes=[0])

    # Visualize results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()