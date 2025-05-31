from pathlib import Path
import blobconverter
import cv2
import depthai
import numpy as np

# Crear el pipeline
pipeline = depthai.Pipeline()

# Nodo de cámara RGB
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)


# Nodo de red neuronal (MobileNetSSD)
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

# Nodo de salida de imagen
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

# Nodo de salida de resultados de detección
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

# Nodo de salida de la IMU
xout_imu = pipeline.createXLinkOut()
xout_imu.setStreamName("imu")

# Nodo de la IMU
imu = pipeline.createIMU()
imu.enableIMUSensor(depthai.IMUSensor.ACCELEROMETER_RAW, 500)
imu.enableIMUSensor(depthai.IMUSensor.GYROSCOPE_RAW, 400)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

# Enlaces
cam_rgb.preview.link(xout_rgb.input)
cam_rgb.preview.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)
imu.out.link(xout_imu.input)

# Ejecutar el dispositivo en modo USB2
with depthai.Device(pipeline, usb2Mode=True) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    q_imu = device.getOutputQueue("imu", maxSize=50, blocking=False)

    frame = None
    detections = []

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def printImu(imuPackets):
        for imuPacket in imuPackets:
            accel = imuPacket.acceleroMeter
            gyro = imuPacket.gyroscope
            print(f"[IMU] Accel [m/s^2]: x={accel.x:.2f}, y={accel.y:.2f}, z={accel.z:.2f}")
            print(f"[IMU] Gyro [rad/s]: x={gyro.x:.2f}, y={gyro.y:.2f}, z={gyro.z:.2f}")

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        in_imu = q_imu.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            detections = in_nn.detections

        if in_imu is not None:
            printImu(in_imu.packets)

        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord('q'):
            break

