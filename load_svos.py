import os
import numpy as np
import cv2
import pyzed.sl as sl

def list_svos(data_dir):
    """List all SVO files in the given folder."""
    svo_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".svo2")]
    svo_files.sort()
    if not svo_files:
        print(f"No SVO files found in {data_dir}")
        return []
    print("\nAvailable SVO recordings:")
    for i, f in enumerate(svo_files, 1):
        print(f"  {i}. {f}")
    return svo_files

def load_from_svo(svo_path):
    """
    Load averaged RGB, Depth, Intrinsics, and Confidence from an SVO file.
    Returns:
        img  (np.uint8, HxWx3)   – same as cv2.imread() RGB image
        depth (np.float32, HxW)  – same as np.load(depth.npy)
        intr  (np.float64, 4,)   – [fx, fy, cx, cy]
        conf  (np.float32, HxW)  – same as np.load(confidence.npy)
    """
    print(f"\nOpening SVO: {svo_path}")

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.1  # meters
    init_params.depth_maximum_distance = 0.5  # meters

    cam = sl.Camera()
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {status}")
        

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    confidence = sl.Mat()
    sensordata = sl.SensorsData()

    n_frames = int(cam.get_svo_number_of_frames())
    n_frames = 1
    print(f"Total frames in SVO: {n_frames}")

    rgb_accum = None
    depth_accum = None
    conf_accum = None
    valid_frames = 0

    for i in range(n_frames):
        print(f'nfrmes is {n_frames}')
        if cam.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            cam.get_sensors_data(sensordata, sl.TIME_REFERENCE.IMAGE)
            acceleration = sensordata.get_imu_data().get_linear_acceleration()
            accel_vec = np.array(acceleration)
            g_imu = -accel_vec / np.linalg.norm(accel_vec)
            cam.retrieve_image(image, sl.VIEW.LEFT)
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            cam.retrieve_measure(confidence, sl.MEASURE.CONFIDENCE)

            rgb = np.asarray(image.get_data())
            if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]
            rgb_np = image.get_data()[:, :, :3]      # drop alpha channel
        
            depth_np = np.asarray(depth.get_data())
            depth_check = depth_np[np.isfinite(depth_np)]
            conf_np = np.asarray(confidence.get_data())

            if rgb_accum is None:
                rgb_accum = np.zeros_like(rgb_np)
                depth_accum = np.zeros_like(depth_np)
                conf_accum = np.zeros_like(conf_np)

            rgb_accum += rgb_np
            depth_accum += depth_np
            conf_accum += conf_np
            valid_frames += 1
        else:
            break

    if valid_frames == 0:
        raise RuntimeError("No valid frames could be read from SVO.")

    # Average everything
    img = np.clip(rgb_accum / valid_frames, 0, 255)
    print(img.shape)
    depth_avg = (depth_accum / valid_frames)
    conf_avg = (conf_accum / valid_frames)

    # Extract intrinsics
    cam_info = cam.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left_cam = calib.left_cam
    intr = np.array([left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy], dtype=np.float64)

    cam.close()
    print(f"Averaged over {valid_frames} frames successfully.")
    img = np.clip(img, 0, 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # convert to BGR if needed
    # cv2.imshow("RGB from Loader", img)
    # Ensure exact format match with old loader
    return img, depth_avg, intr, conf_avg, g_imu