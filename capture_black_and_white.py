"""
zed_capture.py

Capture RGB images, depth maps, and camera intrinsics from ZED camera.
Now includes live black-and-white depth display with invalid pixels marked in red.
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
    init.camera_fps = 15
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NEURAL

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Unable to open ZED camera:", status)
        cam.close()
        return None

    print("ZED opened. Press 'd' to capture a frame, or 'q' to exit.")

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    win_name = "ZED Live (RGB + Depth)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    saved_files = None
    try:
        while True:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image, sl.VIEW.LEFT)
                cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

                img = np.asarray(image.get_data())
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                depth_arr = np.asarray(depth.get_data())

                # --- Create grayscale depth visualization ---
                depth_vis = depth_arr.copy()
                invalid_mask = ~np.isfinite(depth_vis) | (depth_vis <= 0)
                valid_depths = depth_vis[np.isfinite(depth_vis) & (depth_vis > 0)]

                if valid_depths.size > 0:
                    dmin, dmax = np.percentile(valid_depths, [1, 99])
                    depth_vis = np.clip(depth_vis, dmin, dmax)
                else:
                    dmin, dmax = 0, 1

                depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                # Mark invalid depth pixels in red
                depth_bgr[invalid_mask] = (0, 0, 255)

                # Combine RGB + depth views
                disp_rgb = cv2.resize(img, (960, 540))
                disp_depth = cv2.resize(depth_bgr, (960, 540))
                combined = np.hstack((disp_rgb, disp_depth))

                cv2.putText(combined, "Left: RGB | Right: Depth (invalid = red)", 
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined, "Press 'd' to save frame, 'q' to quit", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow(win_name, combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    ts = int(time.time())
                    rgb_path = os.path.join(CAPTURE_DIR, f"rgb_{ts}.png")
                    depth_path = os.path.join(CAPTURE_DIR, f"depth_{ts}.npy")
                    intr_path = os.path.join(CAPTURE_DIR, f"intrinsics_{ts}.npy")

                    cv2.imwrite(rgb_path, img)
                    np.save(depth_path, depth_arr)

                    cam_info = cam.get_camera_information()
                    calib = cam_info.camera_configuration.calibration_parameters
                    left_cam = calib.left_cam
                    fx, fy, cx, cy = left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy
                    np.save(intr_path, np.array([fx, fy, cx, cy], dtype=np.float64))

                    print(f"\nSaved:")
                    print(f"  RGB:   {rgb_path}")
                    print(f"  Depth: {depth_path}")
                    print(f"  Intr:  {intr_path}")
                    saved_files = (rgb_path, depth_path, intr_path)

                elif key == ord('q'):
                    break
            else:
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        cam.close()

    return saved_files


def main():
    print("Starting ZED capture tool with invalid depth visualization.")
    saved = capture_frame_and_save()
    if saved is None:
        print("No capture saved. Exiting.")
    else:
        print(f"\n✅ Capture complete! Files saved in '{CAPTURE_DIR}'.")
        print("Use zed_segment_pointcloud.py or zed_distance_measure.py next.")


if __name__ == '__main__':
    main()
