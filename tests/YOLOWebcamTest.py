from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO preentrenado
model = YOLO('yolo11n.pt')  # Puedes cambiar a yolov8s.pt, etc.

# Abrir la camara (0 = webcam por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la camara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hacer prediccion en el frame de la webcam
    results = model(frame)

    # Dibujar los resultados sobre el frame
    annotated_frame = results[0].plot()

    # Mostrar en pantalla
    cv2.imshow("Deteccion de personas (YOLO)", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

