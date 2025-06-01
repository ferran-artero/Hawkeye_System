#!/usr/bin/env python3
"""
IMU data synchronization (accelerometer and gyroscope) with camera frames
using DepthAI. Includes two methods: nearest neighbor search and linear interpolation.
"""

import depthai as dai
import cv2
import numpy as np
from collections import deque
import threading
import time
from datetime import datetime, timedelta



class IMUCameraSynchronizer:
    """Class for synchronizing IMU data with camera frames"""
    
    def __init__(self, time_tolerance_ms=33, buffer_size=1000):
        """
        Args:
            time_tolerance_ms: Time tolerance in milliseconds (33ms = ~1 frame at 30fps)
            buffer_size: Maximum size of the circular IMU buffer
        """
        self.imu_buffer = deque(maxlen=buffer_size)
        self.time_tolerance_us = time_tolerance_ms * 1000  # Convert to microseconds
        self.lock = threading.Lock()
        
    def add_imu_data(self, imu_packet):
        """Adds IMU data to the circular buffer"""
        with self.lock:
            for imu_data in imu_packet.packets:
                # Timestamp from accelerometer (you can average with gyroscope timestamp for more precision)
                timestamp = imu_data.acceleroMeter.getTimestampDevice().total_seconds()


                # Extract accelerometer data
                accel = imu_data.acceleroMeter
                accel_data = [accel.x, accel.y, accel.z]
                
                # Extract gyroscope data
                gyro = imu_data.gyroscope
                gyro_data = [gyro.x, gyro.y, gyro.z]

                

                # Add to buffer
                self.imu_buffer.append({
                    'timestamp': timestamp,
                    'accel': accel_data,
                    'gyro': gyro_data
                })
    
    def get_synchronized_imu(self, frame_timestamp):
        """
        Gets the IMU data temporally closest to the frame
        
        Args:
            frame_timestamp: Camera frame timestamp
            
        Returns:
            dict: Synchronized IMU data or None if no valid data
        """
        with self.lock:
            if not self.imu_buffer:
                return None
            
            best_match = None
            min_time_diff = float('inf')
            
            # Search for the temporally closest data
            for imu_data in self.imu_buffer:
                time_diff_us = abs(imu_data['timestamp'] * 1e6 - frame_timestamp * 1e6)

                if time_diff_us < min_time_diff:
                    min_time_diff = time_diff_us
                    best_match = imu_data
            
            # Check if it's within the tolerance threshold
            if best_match and min_time_diff <= self.time_tolerance_us:
                return {
                    **best_match,
                    'time_diff_ms': min_time_diff / 1000,  # Convert to ms
                    'method': 'nearest'
                }
            
            return None
    
    def interpolate_imu(self, frame_timestamp):
        """
        Linearly interpolates IMU data for the exact frame timestamp
        """
        with self.lock:
            if len(self.imu_buffer) < 2:
                return None
            
            ts = frame_timestamp
            """
            print(f"[DEBUG] Frame ts: {frame_timestamp:.6f}")
            print(f"[DEBUG] First IMU ts: {self.imu_buffer[0]['timestamp']:.6f}")
            print(f"[DEBUG] Last IMU ts: {self.imu_buffer[-1]['timestamp']:.6f}")
            """

            for i in range(len(self.imu_buffer) - 1):
                d1 = self.imu_buffer[i]
                d2 = self.imu_buffer[i + 1]
                t1 = d1['timestamp']
                t2 = d2['timestamp']


                if t1 <= ts <= t2:
                    # Interpolate between d1 and d2
                    total = t2 - t1
                    if total == 0:
                        continue
                    alpha = (ts - t1) / total

                    interp_accel = [
                        d1['accel'][j] + alpha * (d2['accel'][j] - d1['accel'][j]) for j in range(3)
                    ]
                    interp_gyro = [
                        d1['gyro'][j] + alpha * (d2['gyro'][j] - d1['gyro'][j]) for j in range(3)
                    ]

                    return {
                        'timestamp': frame_timestamp,
                        'accel': interp_accel,
                        'gyro': interp_gyro,
                        'interpolation_factor': alpha,
                        'method': 'interpolated',
                        'before_timestamp': d1['timestamp'],
                        'after_timestamp': d2['timestamp']
                    }

            return None


    
    def get_buffer_stats(self):
        """Gets IMU buffer statistics"""
        with self.lock:
            if not self.imu_buffer:
                return {'count': 0, 'time_span_ms': 0, 'frequency_hz': 0}
            
            oldest_time = self.imu_buffer[0]['timestamp']
            newest_time = self.imu_buffer[-1]['timestamp']

            time_span_s = newest_time - oldest_time
            
            return {
                'count': len(self.imu_buffer),
                'time_span_ms': time_span_s * 1000,
                'frequency_hz': len(self.imu_buffer) / time_span_s if time_span_s > 0 else 0
            }


def create_pipeline():
    """Creates the DepthAI pipeline with camera and IMU"""
    pipeline = dai.Pipeline()
    
    # Configure RGB camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # Changed for OV9782
    cam_rgb.setFps(30)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    # Configure IMU
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor([
        dai.IMUSensor.ACCELEROMETER_RAW,
        dai.IMUSensor.GYROSCOPE_RAW
    ], 100)  # 100 Hz
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    
    # Configure outputs
    cam_out = pipeline.create(dai.node.XLinkOut)
    cam_out.setStreamName("rgb")
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.preview.link(cam_out.input)

    imu_out = pipeline.create(dai.node.XLinkOut)
    imu_out.setStreamName("imu")
    imu.out.link(imu_out.input)
    
    return pipeline


def process_frame_with_imu(frame, imu_data, frame_count):
    """
    Processes the frame with synchronized IMU data
    
    Args:
        frame: Camera frame (numpy array)
        imu_data: Synchronized IMU data
        frame_count: Current frame number
    """
    if imu_data is None:
        return frame
    
    # Create a copy of the frame to draw information
    display_frame = frame.copy()
    
    # Information to display
    info_lines = [
        f"Frame: {frame_count}",
        f"Method: {imu_data.get('method', 'N/A')}",
        f"Accel X: {imu_data['accel'][0]:.3f}",
        f"Accel Y: {imu_data['accel'][1]:.3f}",
        f"Accel Z: {imu_data['accel'][2]:.3f}",
        f"Gyro X: {imu_data['gyro'][0]:.3f}",
        f"Gyro Y: {imu_data['gyro'][1]:.3f}",
        f"Gyro Z: {imu_data['gyro'][2]:.3f}"
    ]
    
    # Add method-specific information
    if imu_data.get('method') == 'nearest':
        info_lines.append(f"Diff: {imu_data.get('time_diff_ms', 0):.1f}ms")
    elif imu_data.get('method') == 'interpolated':
        info_lines.append(f"Factor: {imu_data.get('interpolation_factor', 0):.3f}")
    
    # Draw information on the frame
    y_offset = 30
    for line in info_lines:
        cv2.putText(display_frame, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
    
    return display_frame


def main():
    """Main function with robust error handling"""
    print("Starting IMU-Camera synchronization...")
    
    # Create synchronizer
    synchronizer = IMUCameraSynchronizer(time_tolerance_ms=33, buffer_size=1000)
    
    # Variables for statistics
    frame_count = 0
    sync_success_count = 0
    start_time_monotonic = time.monotonic()
    device = None
    imu_worker = None
    
    try:
        # Search for available devices
        print("Searching for DepthAI devices...")
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("No DepthAI devices found connected")
            return
        
        print(f"Devices found: {len(devices)}")
        for i, device_info in enumerate(devices):
            print(f"  [{i}] {device_info.getMxId()} - {device_info.desc.protocol}")
        
        # Create pipeline
        pipeline = create_pipeline()
        
        # Connect to device with timeout
        print("Connecting to device...")
        with dai.Device(pipeline) as device:

            
            # Verify connection
            if not device.isClosed():
                print("Device connected successfully")
            else:
                print("Error: Could not connect to device")
                return
                
            # Create data queues with robust configuration
            q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)  # Reduced to avoid overflow
            q_imu = device.getOutputQueue("imu", maxSize=20, blocking=False)
            
            print("Queues created. Press 'q' to quit.")
            print("Methods: 'n' = nearest, 'i' = interpolated")
            
            # Flag to control threads
            running = threading.Event()
            running.set()
            
            # Thread to process IMU data with error handling
            def imu_thread():
                consecutive_errors = 0
                max_consecutive_errors = 10
                
                while running.is_set():
                    try:
                        if device.isClosed():
                            print("Device closed, exiting IMU thread")
                            break
                            
                        imu_data = q_imu.tryGet()  # Non-blocking
                        if imu_data is not None:
                            synchronizer.add_imu_data(imu_data)
                            consecutive_errors = 0  # Reset error counter
                        else:
                            time.sleep(0.001)  # Small pause if no data
                            
                    except Exception as e:
                        consecutive_errors += 1
                        if consecutive_errors < max_consecutive_errors:
                            print(f"Error in IMU thread (#{consecutive_errors}): {e}")
                            time.sleep(0.1)  # Pause before retrying
                        else:
                            print(f"Too many consecutive errors in IMU. Exiting thread.")
                            running.clear()
                            break
            
            # Start IMU thread
            imu_worker = threading.Thread(target=imu_thread, daemon=True)
            imu_worker.start()
            
            # Synchronization method (default interpolation)
            use_interpolation = False
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            # Main loop with robust error handling
            while running.is_set():
                try:
                    # Check device status
                    if device.isClosed():
                        print("Device disconnected")
                        break
                    
                    # Get camera frame non-blocking
                    in_rgb = q_rgb.tryGet()
                    if in_rgb is None:
                        time.sleep(0.001)  # Small pause if no frames
                        continue
                    
                    frame_count += 1
                    frame = in_rgb.getCvFrame()
                    frame_timestamp = in_rgb.getTimestampDevice().total_seconds()



                    
                    # Verify frame is valid
                    if frame is None or frame.size == 0:
                        print("Invalid frame received")
                        continue
                    
                    # Synchronize with IMU
                    synced_imu = None
                    try:
                        if use_interpolation:
                            synced_imu = synchronizer.interpolate_imu(frame_timestamp)
                        else:
                            synced_imu = synchronizer.get_synchronized_imu(frame_timestamp)
                    except Exception as e:
                        print(f"Error in synchronization: {e}")
                    
                    if synced_imu:
                        sync_success_count += 1
                    
                    # Process frame with IMU data
                    try:
                        display_frame = process_frame_with_imu(frame, synced_imu, frame_count)
                        
                        # Show frame
                        cv2.imshow("IMU-Camera Sync", display_frame)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue
                    
                    # Show statistics every 30 frames
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time_monotonic
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        sync_rate = (sync_success_count / frame_count) * 100 if frame_count > 0 else 0
                        
                        try:
                            buffer_stats = synchronizer.get_buffer_stats()
                            print(f"\n--- Statistics (Frame {frame_count}) ---")
                            print(f"FPS: {fps:.1f}")
                            print(f"Successful synchronization: {sync_rate:.1f}%")
                            print(f"IMU Buffer: {buffer_stats['count']} samples")
                            print(f"IMU Frequency: {buffer_stats['frequency_hz']:.1f} Hz")
                            print(f"Current method: {'Interpolation' if use_interpolation else 'Nearest'}")
                        except Exception as e:
                            print(f"Error showing statistics: {e}")
                    
                    # Handle keys
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Exiting by user request...")
                        break
                    elif key == ord('i'):
                        use_interpolation = True
                        print("Switched to interpolation method")
                    elif key == ord('n'):
                        use_interpolation = False
                        print("Switched to nearest method")
                    elif key == ord('r'):
                        # Reset statistics
                        frame_count = 0
                        sync_success_count = 0
                        start_time_monotonic = time.monotonic()

                        print("Statistics reset")
                    
                    consecutive_errors = 0  # Reset counter if everything is fine
                    
                except KeyboardInterrupt:
                    print("\nKeyboard interrupt")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error in main loop (#{consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors. Exiting...")
                        break
                    else:
                        time.sleep(0.1)  # Pause before continuing
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Safe cleanup
        print("Cleaning up resources...")
        
        if 'running' in locals():
            running.clear()
        
        if imu_worker and imu_worker.is_alive():
            print("Waiting for IMU thread...")
            imu_worker.join(timeout=2.0)
        
        if device and not device.isClosed():
            print("Closing device...")
            try:
                device.close()
            except:
                pass
        
        cv2.destroyAllWindows()
        
        print(f"\nFinal summary:")
        print(f"Frames processed: {frame_count}")
        print(f"Successful synchronizations: {sync_success_count}")
        if frame_count > 0:
            print(f"Success rate: {(sync_success_count/frame_count)*100:.1f}%")
        print("Program terminated")


if __name__ == "__main__":
    main()
