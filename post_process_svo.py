import pyzed.sl as sl
import cv2
import os
import glob
from capture_frames import extract_image_metrics
import numpy as np

def list_svo_files(participant_id, base_dir="robot_data"):
    """List all .svo2 files for the given participant ID."""
    search_path = os.path.join(base_dir, f"{participant_id}_data_robotic", str(participant_id), "*.svo2")
    return sorted(glob.glob(search_path))

def browse_svo_file(svo_path, output_folder):
    """Open one .svo2 file and let user browse & optionally save a frame."""
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.camera_fps = 30
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.1  # meters
    init.depth_maximum_distance = 0.4  # meters
    zed = sl.Camera()

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open {svo_path}")
        return False  # Skip to next file

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    confidence = sl.Mat()
    sensordata = sl.SensorsData()
    total_frames = zed.get_svo_number_of_frames()
    current_frame = 0

    print(f"\nOpened {os.path.basename(svo_path)} with {total_frames} frames.")
    print("Use ← / → to navigate, SPACE to save frame, ESC to quit this file.")

    while True:
        zed.set_svo_position(current_frame)
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.get_sensors_data(sensordata, sl.TIME_REFERENCE.IMAGE)
            acceleration = sensordata.get_imu_data().get_linear_acceleration()
            accel_vec = np.array(acceleration)
            # save intrinsics
            cam_info = zed.get_camera_information()
            calib = cam_info.camera_configuration.calibration_parameters
            left_cam = calib.left_cam
            intrinsics = (left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy)
            fx, fy, cx, cy = intrinsics
            # ---------------------gravity direction----------------------
            # In static condition (no motion), -accel ≈ gravity in IMU frame
            g_imu = -accel_vec / np.linalg.norm(accel_vec)
            imu_pose = sensordata.get_imu_data().get_pose()
            R = imu_pose.get_rotation_matrix().r
            g_cam = R.T @ g_imu
            print("Gravity dir (camera frame) from raw accel:", g_cam/np.linalg.norm(g_cam))
            print("Gravity dir (world frame) from raw accel:", g_imu/np.linalg.norm(g_imu))
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(confidence, sl.MEASURE.CONFIDENCE)  # Retrieve confidence map
            frame = image.get_data()
            display = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            cv2.putText(display, f"Frame {current_frame+1}/{total_frames}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(display, f"gravity world {g_imu}",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(display, f"gravity camera frame {g_cam}",
                        (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # draw gravity direction for check
            x0 = 10
            y0 = 10
            scale = 0.1  # arbitrary length in meters to make arrow visible
            X = g_imu[0] * scale
            Y = g_imu[1] * scale
            Z = 0.4
            u = int(fx * X / Z + x0)
            v = int(fy * Y / Z + y0)
            print(u,v)
            # Draw arrow
            cv2.arrowedLine(display, (x0, y0), (u, v), (255, 0, 0), 3, tipLength=0.3)
            cv2.imshow("SVO Browser", display)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key in [81, ord('a')]:  # Left
            current_frame = max(0, current_frame - 1)
        elif key in [83, ord('d')]:  # Right
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == 32:  # SPACE
            # Build clean save path (remove old extension first)
            base = os.path.splitext(os.path.basename(svo_path))[0]
            save_name = f"{base}_frame_{current_frame:06d}"
            save_path = os.path.join(output_folder, save_name)

            success = extract_image_metrics(output_folder, save_name, image, depth, confidence, intrinsics, g_imu)
            if success: 
                print(f"Saved frame {current_frame} -> {save_path}")
            else:
                print("Failed to save frame.")

            zed.close()
            cv2.destroyAllWindows()
            return True  # saved -> close current file and return

    zed.close()
    cv2.destroyAllWindows()
    return False  # user quit file without saving

def main():
    participant_id = input("Enter participant ID (e.g., 1): ").strip()
    svo_files = list_svo_files(participant_id)

    if not svo_files:
        print("No .svo2 files found for this participant.")
        return

    print(f"Found {len(svo_files)} .svo2 files for participant {participant_id}.")

    for i, svo_path in enumerate(svo_files):
        print(f"\nProcessing file {i+1}/{len(svo_files)}: {os.path.basename(svo_path)}")
        output_folder = os.path.join("saved_robot_data", f"{participant_id}_data_robotic", f"run{i+1}")
        os.makedirs(output_folder, exist_ok=True)
        saved = browse_svo_file(svo_path, output_folder)

        print("\nContinue press ENTER, or 'q' to quit:")
        user_in = input().strip().lower()
        if user_in == 'q':
            print("Exiting.")
            break

if __name__ == "__main__":
    main()
