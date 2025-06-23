from omni.isaac.core import SimulationContext
from omni.isaac.sensor import Camera
from ultralytics import YOLO
import numpy as np
import cv2
import math

# Inicializar simulación
sim = SimulationContext()
sim.play()
camera = Camera("/World/Camera", frequency=30.0)
camera.initialize()

model = YOLO("yolov8n.pt")

# Variables de tracking
selected_id_index = 0
tracked_ids = []

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def distance_to_center(cx, cy, frame_width, frame_height):
    return math.sqrt((cx - frame_width // 2)**2 + (cy - frame_height // 2)**2)

def draw_info(frame, cx, cy, dist, obj_id):
    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
    cv2.line(frame, (cx, cy), (frame.shape[1]//2, frame.shape[0]//2), (255, 0, 0), 2)
    cv2.putText(frame, f"ID: {obj_id} Dist: {int(dist)} px", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

while sim.is_playing():
    sim.step(render=True)  # Actualizar simulación

    # Obtener imagen RGB de la cámara virtual
    frame = camera.get_rgba()
    frame = (frame[:, :, :3] * 255).astype(np.uint8)  # Quitar alpha y convertir a uint8 para OpenCV

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, classes=[0])

    tracked_ids = []
    centers = {}

    if len(results) > 0:
        for r in results:
            if r.boxes is None or r.boxes.id is None or r.boxes.xyxy is None:
                continue
            for box, obj_id in zip(r.boxes.xyxy, r.boxes.id):
                tracked_ids.append(int(obj_id))
                centers[int(obj_id)] = get_center(box.cpu().numpy())

    if len(tracked_ids) == 0:
        selected_id_index = 0
    else:
        selected_id_index = max(0, min(selected_id_index, len(tracked_ids) - 1))
        current_id = tracked_ids[selected_id_index]
        cx, cy = centers[current_id]
        dist = distance_to_center(cx, cy, frame.shape[1], frame.shape[0])
        dx = cx - frame.shape[1] // 2
        dy = cy - frame.shape[0] // 2
        print(f"Vector al centro: dx={dx}, dy={dy}, distancia escalar: {dist}")
        draw_info(frame, cx, cy, dist, current_id)

    annotated_frame = results[0].plot()
    combined = cv2.addWeighted(frame, 0.7, annotated_frame, 0.3, 0)
    cv2.imshow("YOLOv8 Tracking in Isaac Sim", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        selected_id_index = max(0, selected_id_index - 1)
    elif key == ord('d'):
        selected_id_index = min(len(tracked_ids) - 1, selected_id_index + 1)

cv2.destroyAllWindows()