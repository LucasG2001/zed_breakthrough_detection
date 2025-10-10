"""
zed_capture.py

Capture RGB images, depth maps, and camera intrinsics from ZED camera.
"""

import os
import sys
import time
import numpy as np
import cv2

try:
    import pyzed.sl as sl
except Exception as e:
    print("Failed to import ZED SDK Python bindings (pyzed.sl). Make sure ZED SDK and Python API are installed.")
    print("Error:", e)
    sys.exit(1)


CAPTURE_DIR = "zed_captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)


def capture_frame_and_save():
    init = sl.InitParameters()
    init.camera_fps = 30
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.1  # meters
    init.depth_maximum_distance = 0.5  # meters


    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Unable to open ZED camera:", status)
        cam.close()
        return None

    print("ZED opened. Press 'd' in the live window to capture a frame, or 'q' to exit.")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 100
    
    image = sl.Mat()
    depth = sl.Mat()
    confidence = sl.Mat()  # Add confidence map

    win_name = "ZED Live"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    saved_files = None
    try:
        while True:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image, sl.VIEW.LEFT)
                cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
                cam.retrieve_measure(confidence, sl.MEASURE.CONFIDENCE)  # Retrieve confidence map

                img = np.asarray(image.get_data())
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                depth_arr = np.asarray(depth.get_data())
                confidence_arr = np.asarray(confidence.get_data())  # Convert confidence map to numpy

                # --- Depth visualization ---
                depth_disp = np.copy(depth_arr)
                depth_disp[~np.isfinite(depth_disp)] = 0
                valid = depth_disp > 0
                if np.any(valid):
                    min_z = np.percentile(depth_disp[valid], 2)
                    max_z = np.percentile(depth_disp[valid], 98)
                else:
                    min_z, max_z = 0, 1
                depth_norm = np.clip((depth_disp - min_z) / (max_z - min_z + 1e-6), 0, 1)
                depth_vis = (depth_norm * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                depth_vis = cv2.resize(depth_vis, (img.shape[1], img.shape[0]))

                # --- Stack RGB and depth for display ---
                preview = np.hstack([img, depth_vis])

                disp = preview.copy()
                cv2.putText(disp, "Press 'd' to save frame, 'q' to quit", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(win_name, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    ts = int(time.time())
                    rgb_path = os.path.join(CAPTURE_DIR, f"rgb_{ts}.png")
                    depth_path = os.path.join(CAPTURE_DIR, f"depth_{ts}.npy")
                    intr_path = os.path.join(CAPTURE_DIR, f"intrinsics_{ts}.npy")
                    conf_path = os.path.join(CAPTURE_DIR, f"confidence_{ts}.npy")

                    cv2.imwrite(rgb_path, img)
                    np.save(depth_path, depth_arr)
                    np.save(conf_path, confidence_arr)

                    # Display confidence as grayscale: white = high confidence, black = low
                    conf_gray = np.clip(confidence_arr, 0, 100)
                    conf_gray = (conf_gray * 255 / 100).astype(np.uint8)
                    cv2.imshow("Confidence (Grayscale)", conf_gray)
                    print("Confidence map displayed. Press any key to continue...")
                    cv2.waitKey(0)
                    cv2.destroyWindow("Confidence (Grayscale)")
                    cam_info = cam.get_camera_information()
                    calib = cam_info.camera_configuration.calibration_parameters
                    left_cam = calib.left_cam
                    fx, fy, cx, cy = left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy
                    np.save(intr_path, np.array([fx, fy, cx, cy], dtype=np.float64))

                    print(f"Saved: {rgb_path}, {depth_path}, {intr_path}")
                    saved_files = (rgb_path, depth_path, intr_path)

                elif key == ord('q'):
                    break
            else:
                time.sleep(0.01)
    finally:
        cv2.destroyWindow(win_name)
        cam.close()

    return saved_files


def main():
    print("Starting ZED capture tool")
    saved = capture_frame_and_save()
    if saved is None:
        print("No capture saved. Exiting.")
    else:
        print(f"\nCapture complete! Files saved in '{CAPTURE_DIR}' directory.")
        print("Use zed_segment.py to segment and visualize 3D points.")


if __name__ == '__main__':
    main()