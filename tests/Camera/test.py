#!/usr/bin/env python3
"""
Sincronización de datos de IMU (acelerómetro y giroscopio) con frames de cámara
usando DepthAI. Incluye dos métodos: búsqueda del más cercano e interpolación lineal.
"""

import depthai as dai
import cv2
import numpy as np
from collections import deque
import threading
import time
from datetime import datetime, timedelta



class IMUCameraSynchronizer:
    """Clase para sincronizar datos de IMU con frames de cámara"""
    
    def __init__(self, time_tolerance_ms=33, buffer_size=1000):
        """
        Args:
            time_tolerance_ms: Tolerancia temporal en milisegundos (33ms = ~1 frame a 30fps)
            buffer_size: Tamaño máximo del buffer circular de IMU
        """
        self.imu_buffer = deque(maxlen=buffer_size)
        self.time_tolerance_us = time_tolerance_ms * 1000  # Convertir a microsegundos
        self.lock = threading.Lock()
        
    def add_imu_data(self, imu_packet):
        """Añade datos de IMU al buffer circular"""
        with self.lock:
            for imu_data in imu_packet.packets:
                # Timestamp desde acelerómetro (puedes hacer una media con el del giroscopio si quieres más precisión)
                timestamp = imu_data.acceleroMeter.getTimestampDevice()

                # Extraer datos del acelerómetro
                accel = imu_data.acceleroMeter
                accel_data = [accel.x, accel.y, accel.z]
                
                # Extraer datos del giroscopio
                gyro = imu_data.gyroscope
                gyro_data = [gyro.x, gyro.y, gyro.z]

                

                # Añadir al buffer
                self.imu_buffer.append({
                    'timestamp': timestamp,
                    'accel': accel_data,
                    'gyro': gyro_data
                })
    
    def get_synchronized_imu(self, frame_timestamp):
        """
        Obtiene el dato de IMU más cercano temporalmente al frame
        
        Args:
            frame_timestamp: Timestamp del frame de cámara
            
        Returns:
            dict: Datos de IMU sincronizados o None si no hay datos válidos
        """
        with self.lock:
            if not self.imu_buffer:
                return None
            
            best_match = None
            min_time_diff = float('inf')
            
            # Buscar el dato más cercano temporalmente
            for imu_data in self.imu_buffer:
                time_diff_us = abs(imu_data['timestamp'].total_seconds() * 1000000 - 
                                 frame_timestamp.total_seconds() * 1000000)
                if time_diff_us < min_time_diff:
                    min_time_diff = time_diff_us
                    best_match = imu_data
            
            # Verificar si está dentro del umbral de tolerancia
            if best_match and min_time_diff <= self.time_tolerance_us:
                return {
                    **best_match,
                    'time_diff_ms': min_time_diff / 1000,  # Convertir a ms
                    'method': 'nearest'
                }
            
            return None
    
    def interpolate_imu(self, frame_timestamp):
        """
        Interpola linealmente los datos de IMU para el timestamp exacto del frame
        
        Args:
            frame_timestamp: Timestamp del frame de cámara
            
        Returns:
            dict: Datos de IMU interpolados o None si no es posible interpolar
        """
        with self.lock:
            if len(self.imu_buffer) < 2:
                return None
            
            # Buscar los dos puntos para interpolación (antes y después)
            before = None
            after = None
            
            for imu_data in self.imu_buffer:
                if (imu_data['timestamp'].total_seconds() <= 
                    frame_timestamp.total_seconds()):
                    before = imu_data
                elif (imu_data['timestamp'].total_seconds() > 
                      frame_timestamp.total_seconds() and after is None):
                    after = imu_data
                    break
            
            # Si tenemos ambos puntos, interpolar
            if before and after:
                # Calcular factor de interpolación (0.0 a 1.0)
                total_time = (after['timestamp'].total_seconds() - 
                            before['timestamp'].total_seconds())
                elapsed_time = (frame_timestamp.total_seconds() - 
                              before['timestamp'].total_seconds())
                
                if total_time > 0:
                    t = elapsed_time / total_time
                
                if total_time > 0:
                    t = elapsed_time / total_time
                    
                    # Interpolación lineal para acelerómetro
                    interpolated_accel = [
                        before['accel'][i] + t * (after['accel'][i] - before['accel'][i])
                        for i in range(3)
                    ]
                    
                    # Interpolación lineal para giroscopio
                    interpolated_gyro = [
                        before['gyro'][i] + t * (after['gyro'][i] - before['gyro'][i])
                        for i in range(3)
                    ]
                    
                    return {
                        'timestamp': frame_timestamp,
                        'accel': interpolated_accel,
                        'gyro': interpolated_gyro,
                        'interpolation_factor': t,
                        'method': 'interpolated',
                        'before_timestamp': before['timestamp'],
                        'after_timestamp': after['timestamp']
                    }
            
            return None
    
    def get_buffer_stats(self):
        """Obtiene estadísticas del buffer de IMU"""
        with self.lock:
            if not self.imu_buffer:
                return {'count': 0, 'time_span_ms': 0, 'frequency_hz': 0}
            
            oldest_time = self.imu_buffer[0]['timestamp'].total_seconds()
            newest_time = self.imu_buffer[-1]['timestamp'].total_seconds()
            time_span_s = newest_time - oldest_time
            
            return {
                'count': len(self.imu_buffer),
                'time_span_ms': time_span_s * 1000,
                'frequency_hz': len(self.imu_buffer) / time_span_s if time_span_s > 0 else 0
            }


def create_pipeline():
    """Crea el pipeline de DepthAI con cámara e IMU"""
    pipeline = dai.Pipeline()
    
    # Configurar cámara RGB
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # Cambiado para OV9782
    cam_rgb.setFps(30)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    # Configurar IMU
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor([
        dai.IMUSensor.ACCELEROMETER_RAW,
        dai.IMUSensor.GYROSCOPE_RAW
    ], 100)  # 100 Hz
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    
    # Configurar salidas
    cam_out = pipeline.create(dai.node.XLinkOut)
    cam_out.setStreamName("rgb")
    cam_rgb.video.link(cam_out.input)
    
    imu_out = pipeline.create(dai.node.XLinkOut)
    imu_out.setStreamName("imu")
    imu.out.link(imu_out.input)
    
    return pipeline


def process_frame_with_imu(frame, imu_data, frame_count):
    """
    Procesa el frame con los datos de IMU sincronizados
    
    Args:
        frame: Frame de la cámara (numpy array)
        imu_data: Datos de IMU sincronizados
        frame_count: Número del frame actual
    """
    if imu_data is None:
        return frame
    
    # Crear una copia del frame para dibujar información
    display_frame = frame.copy()
    
    # Información a mostrar
    info_lines = [
        f"Frame: {frame_count}",
        f"Método: {imu_data.get('method', 'N/A')}",
        f"Accel X: {imu_data['accel'][0]:.3f}",
        f"Accel Y: {imu_data['accel'][1]:.3f}",
        f"Accel Z: {imu_data['accel'][2]:.3f}",
        f"Gyro X: {imu_data['gyro'][0]:.3f}",
        f"Gyro Y: {imu_data['gyro'][1]:.3f}",
        f"Gyro Z: {imu_data['gyro'][2]:.3f}"
    ]
    
    # Agregar información específica del método
    if imu_data.get('method') == 'nearest':
        info_lines.append(f"Diff: {imu_data.get('time_diff_ms', 0):.1f}ms")
    elif imu_data.get('method') == 'interpolated':
        info_lines.append(f"Factor: {imu_data.get('interpolation_factor', 0):.3f}")
    
    # Dibujar información en el frame
    y_offset = 30
    for line in info_lines:
        cv2.putText(display_frame, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
    
    return display_frame


def main():
    """Función principal con manejo robusto de errores"""
    print("Iniciando sincronización IMU-Cámara...")
    
    # Crear sincronizador
    synchronizer = IMUCameraSynchronizer(time_tolerance_ms=33, buffer_size=1000)
    
    # Variables para estadísticas
    frame_count = 0
    sync_success_count = 0
    start_time = time.time()
    device = None
    imu_worker = None
    
    try:
        # Buscar dispositivos disponibles
        print("Buscando dispositivos DepthAI...")
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("No se encontraron dispositivos DepthAI conectados")
            return
        
        print(f"Dispositivos encontrados: {len(devices)}")
        for i, device_info in enumerate(devices):
            print(f"  [{i}] {device_info.getMxId()} - {device_info.desc.protocol}")
        
        # Crear pipeline
        pipeline = create_pipeline()
        
        # Conectar al dispositivo con timeout
        print("Conectando al dispositivo...")
        device = dai.Device(pipeline)
        
        # Verificar conexión
        if not device.isClosed():
            print("Dispositivo conectado exitosamente")
        else:
            print("Error: No se pudo conectar al dispositivo")
            return
            
        # Crear colas de datos con configuración robusta
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)  # Reducido para evitar overflow
        q_imu = device.getOutputQueue("imu", maxSize=20, blocking=False)
        
        print("Colas creadas. Presiona 'q' para salir.")
        print("Métodos: 'n' = nearest, 'i' = interpolated")
        
        # Flag para controlar threads
        running = threading.Event()
        running.set()
        
        # Thread para procesar datos de IMU con manejo de errores
        def imu_thread():
            consecutive_errors = 0
            max_consecutive_errors = 10
            
            while running.is_set():
                try:
                    if device.isClosed():
                        print("Dispositivo cerrado, saliendo del thread IMU")
                        break
                        
                    imu_data = q_imu.tryGet()  # No bloqueante
                    if imu_data is not None:
                        synchronizer.add_imu_data(imu_data)
                        consecutive_errors = 0  # Reset contador de errores
                    else:
                        time.sleep(0.001)  # Pequeña pausa si no hay datos
                        
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors < max_consecutive_errors:
                        print(f"Error en thread IMU (#{consecutive_errors}): {e}")
                        time.sleep(0.1)  # Pausa antes de reintentar
                    else:
                        print(f"Demasiados errores consecutivos en IMU. Saliendo del thread.")
                        running.clear()
                        break
        
        # Iniciar thread de IMU
        imu_worker = threading.Thread(target=imu_thread, daemon=True)
        imu_worker.start()
        
        # Método de sincronización (por defecto interpolación)
        use_interpolation = True
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Loop principal con manejo robusto de errores
        while running.is_set():
            try:
                # Verificar estado del dispositivo
                if device.isClosed():
                    print("Dispositivo desconectado")
                    break
                
                # Obtener frame de cámara de forma no bloqueante
                in_rgb = q_rgb.tryGet()
                if in_rgb is None:
                    time.sleep(0.001)  # Pequeña pausa si no hay frames
                    continue
                
                frame_count += 1
                frame = in_rgb.getCvFrame()
                frame_timestamp = in_rgb.getTimestamp()
                
                # Verificar que el frame es válido
                if frame is None or frame.size == 0:
                    print("Frame inválido recibido")
                    continue
                
                # Sincronizar con IMU
                synced_imu = None
                try:
                    if use_interpolation:
                        synced_imu = synchronizer.interpolate_imu(frame_timestamp)
                    else:
                        synced_imu = synchronizer.get_synchronized_imu(frame_timestamp)
                except Exception as e:
                    print(f"Error en sincronización: {e}")
                
                if synced_imu:
                    sync_success_count += 1
                
                # Procesar frame con datos de IMU
                try:
                    display_frame = process_frame_with_imu(frame, synced_imu, frame_count)
                    
                    # Mostrar frame
                    cv2.imshow("IMU-Camera Sync", display_frame)
                except Exception as e:
                    print(f"Error procesando frame: {e}")
                    continue
                
                # Mostrar estadísticas cada 30 frames
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    sync_rate = (sync_success_count / frame_count) * 100 if frame_count > 0 else 0
                    
                    try:
                        buffer_stats = synchronizer.get_buffer_stats()
                        print(f"\n--- Estadísticas (Frame {frame_count}) ---")
                        print(f"FPS: {fps:.1f}")
                        print(f"Sincronización exitosa: {sync_rate:.1f}%")
                        print(f"Buffer IMU: {buffer_stats['count']} muestras")
                        print(f"Frecuencia IMU: {buffer_stats['frequency_hz']:.1f} Hz")
                        print(f"Método actual: {'Interpolación' if use_interpolation else 'Más cercano'}")
                    except Exception as e:
                        print(f"Error mostrando estadísticas: {e}")
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Saliendo por petición del usuario...")
                    break
                elif key == ord('i'):
                    use_interpolation = True
                    print("Cambiado a método de interpolación")
                elif key == ord('n'):
                    use_interpolation = False
                    print("Cambiado a método del más cercano")
                elif key == ord('r'):
                    # Reset estadísticas
                    frame_count = 0
                    sync_success_count = 0
                    start_time = time.time()
                    print("Estadísticas reiniciadas")
                
                consecutive_errors = 0  # Reset contador si todo va bien
                
            except KeyboardInterrupt:
                print("\nInterrupción por teclado")
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"Error en loop principal (#{consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Demasiados errores consecutivos. Saliendo...")
                    break
                else:
                    time.sleep(0.1)  # Pausa antes de continuar
    
    except Exception as e:
        print(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpieza segura
        print("Limpiando recursos...")
        
        if 'running' in locals():
            running.clear()
        
        if imu_worker and imu_worker.is_alive():
            print("Esperando thread IMU...")
            imu_worker.join(timeout=2.0)
        
        if device and not device.isClosed():
            print("Cerrando dispositivo...")
            try:
                device.close()
            except:
                pass
        
        cv2.destroyAllWindows()
        
        print(f"\nResumen final:")
        print(f"Frames procesados: {frame_count}")
        print(f"Sincronizaciones exitosas: {sync_success_count}")
        if frame_count > 0:
            print(f"Tasa de éxito: {(sync_success_count/frame_count)*100:.1f}%")
        print("Programa terminado")


if __name__ == "__main__":
    main()
