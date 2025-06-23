from ultralytics import YOLO
import cv2
import os
import math

# Cambiar directorio actual
os.chdir(r"c:/Users/ferra/OneDrive/Desktop/thws/Robotics Project - Hawkeye System/Hawkeye_System/src/ObjectTracking")

model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture(1)

# Variables globales para seleccionar ID
selected_id_index = 0
tracked_ids = []

def get_center(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return int(cx), int(cy)

def distance_to_center(cx, cy, frame_width, frame_height):
    center_x = frame_width // 2
    center_y = frame_height // 2
    return math.sqrt((cx - center_x)**2 + (cy - center_y)**2)

def draw_info(frame, cx, cy, dist, obj_id):
    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)  # Punto centro bbox
    cv2.line(frame, (cx, cy), (frame.shape[1]//2, frame.shape[0]//2), (255, 0, 0), 2)  # Línea al centro pantalla
    cv2.putText(frame, f"ID: {obj_id} Dist: {int(dist)} px", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="mybytetrack.yaml", conf=0.5, classes=[0])

    # Obtener IDs detectados en este frame y guardar sus bbox centrales
    tracked_ids = []
    centers = {}
    if len(results) > 0:
        for r in results:
            if r.boxes is None or r.boxes.id is None or r.boxes.xyxy is None:
                continue
            for box, obj_id in zip(r.boxes.xyxy, r.boxes.id):
                tracked_ids.append(int(obj_id))
                centers[int(obj_id)] = get_center(box.cpu().numpy())



    # Evitar error si no hay objetos
    if len(tracked_ids) == 0:
        selected_id_index = 0
    else:
        # Ajustar índice seleccionado para que esté dentro del rango
        selected_id_index = max(0, min(selected_id_index, len(tracked_ids) - 1))

        # Elegir el ID actual
        current_id = tracked_ids[selected_id_index]
        cx, cy = centers[current_id]
        dist = distance_to_center(cx, cy, frame.shape[1], frame.shape[0])

        # Vector al centro
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        dx = cx - center_x
        dy = cy - center_y
        print(f"Vector al centro: dx={dx}, dy={dy}, distancia escalar: {dist}")

        # Dibujar info sobre el objeto seleccionado
        draw_info(frame, cx, cy, dist, current_id)

    annotated_frame = results[0].plot()
    combined = cv2.addWeighted(frame, 0.7, annotated_frame, 0.3, 0)

    cv2.imshow("YOLOv8 Tracking", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):  # Tecla para bajar índice
        selected_id_index = max(0, selected_id_index - 1)
    elif key == ord('d'):  # Tecla para subir índice
        selected_id_index = min(len(tracked_ids) - 1, selected_id_index + 1)

cam.release()
cv2.destroyAllWindows()
