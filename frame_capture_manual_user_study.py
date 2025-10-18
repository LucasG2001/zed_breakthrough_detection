import os
import sys
import time
import random
from datetime import datetime
import numpy as np
try:
    import pyzed.sl as sl
except Exception as e:
    print("Failed to import ZED SDK Python bindings (pyzed.sl). Ensure ZED SDK + Python API are installed.")
    print("Error:", e)
    sys.exit(1)
    
# TODO:test with averaged values

def setup_directories(participant_id, run, angle):
    base_dir = f"manual_data/{participant_id}_data_manual"
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, f"{participant_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def capture_svo(participant_id, run, bone):
    # Randomly choose drill angle
    drill_angle = random.choice([60, 70, 80, 90])
    print(f"Selected drill angle: {drill_angle} degrees")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create directory
    run_dir = setup_directories(participant_id, run, drill_angle)
    svo_path = os.path.join(run_dir, f"Run{run}_{bone}_{drill_angle}_{timestamp}.svo")
    print(f"SVO will be saved to: {svo_path}")
    

    # Initialize ZED parameters
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

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    confidence = sl.Mat()

    win_name = "ZED Live"
    import cv2
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print("Press 'd' to start recording 10 frames, 'q' to quit.")

    try:
        while True:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_measure(confidence, sl.MEASURE.CONFIDENCE)
                confidence_arr = np.asarray(confidence.get_data())  # Convert confidence map to numpy

                # Show preview with angle
                confidence_np = confidence_arr.copy()
                cv2.putText(confidence_np, f"Angle: {drill_angle}°", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(confidence_np, "Press 'd' to record, 'q' to quit", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                conf_disp = 100 - confidence_np
                conf_disp = np.clip(conf_disp, 0, 100)
                conf_norm = (conf_disp / 100.0 * 255).astype(np.uint8)
                cv2.imshow("Confidence (100-confidence, grayscale)", conf_norm)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    print("Recording SVO (10 frames)...")

                    rec_params = sl.RecordingParameters(
                        svo_path,
                        sl.SVO_COMPRESSION_MODE.H265_LOSSLESS
                    )
                    err = cam.enable_recording(rec_params)
                    if err != sl.ERROR_CODE.SUCCESS:
                        print("Failed to enable SVO recording:", err)
                        break

                    frame_count = 0
                    while frame_count < 1:
                        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                            frame_count += 1
                    cam.disable_recording()
                    print(f"Recording finished. {frame_count} frames saved.")
                    break

                elif key == ord('q'):
                    print("User exited.")
                    break

    finally:
        cv2.destroyAllWindows()
        cam.close()

    print(f"SVO saved successfully to {svo_path}")
    print("Camera closed. Exiting.")
    return run_dir


def main():
    participant_id = input("Enter participant ID: ").strip()
    run = input("Enter run number (1, 2, 3...): ").strip()
    bones = ["GENERIC", "FEMUR", "ULNA"]
    bone_id = input("Enter Bone type (GENERIC/FEMUR/ULNA = 0, 1, 2)") 

    if not participant_id or not run:
        print("Participant ID and run number are required.")
        return

    saved_dir = capture_svo(participant_id, run, bones[int(bone_id)])
    if saved_dir:
        print(f"\n✅ Capture complete! Data saved in: {saved_dir}")
    else:
        print("No data saved.")


if __name__ == "__main__":
    main()
