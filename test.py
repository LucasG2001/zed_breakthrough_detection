"""
zed_segment_pointcloud.py (updated)

Automatically compute and visualize 3D point cloud + fitted line right after finalizing polygon (press 'f').
"""

import os
import sys
import time
import numpy as np
import cv2
import open3d as o3d

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

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Unable to open ZED camera:", status)
        cam.close()
        return None

    print("ZED opened. Press 'd' in the live window to capture a frame, or 'q' to exit.")

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    win_name = "ZED Live"
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

                disp = img.copy()
                cv2.putText(disp, "Press 'd' to save frame, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(win_name, disp)

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

class PolySegmenter:
    def __init__(self, image, depth_arr, intr):
        self.image = image.copy()
        self.clone = image.copy()
        self.window = "Segmentation - left-click add points, 'f' finalize, 'r' reset, 'q' quit"
        self.points = []
        self.done = False
        self.depth_arr = depth_arr
        self.intr = intr
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.clone, (x, y), 3, (0, 0, 255), -1)
            if len(self.points) > 1:
                cv2.line(self.clone, self.points[-2], self.points[-1], (0, 255, 0), 2)

    def reset(self):
        self.clone = self.image.copy()
        self.points = []
        self.done = False

    def finalize_mask(self):
        if len(self.points) >= 3:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.done = True
            return mask
        else:
            print("Need at least 3 points to form a polygon")
            return None

    def run(self):
        while True:
            display = self.clone.copy()
            cv2.imshow(self.window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.reset()
            elif key == ord('f'):
                mask = self.finalize_mask()
                if mask is not None:
                    print("Polygon finalized — computing 3D points and visualizing...")
                    pts, colors = mask_to_pointcloud(mask, self.depth_arr, self.intr, self.image)
                    if pts is not None and len(pts) > 1:
                        centroid, direction = fit_line_pca(pts)
                        print("fitted line direction: ", direction)
                        visualize_with_open3d(pts, centroid, direction, colors)
                    else:
                        print("Not enough valid 3D points.")
                    break

            elif key == ord('q'):
                break
        cv2.destroyAllWindows()

def mask_to_pointcloud(mask, depth_arr, intr, rgb_img=None):
    """Backproject masked pixels to 3D. Optionally return per-point colors."""
    fx, fy, cx, cy = intr
    ys, xs = np.where(mask == 255)
    zs = depth_arr[ys, xs]
    valid = np.isfinite(zs) & (zs > 0.01)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    if len(zs) == 0:
        return None, None

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    pts = np.vstack((X, Y, Z)).T

    colors = None
    if rgb_img is not None:
        rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        colors = rgb[ys, xs] / 255.0

    return pts, colors


def fit_line_pca(points):
    centroid = points.mean(axis=0)
    pts_centered = points - centroid
    U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    direction = Vt[0]
    return centroid, direction


def visualize_with_open3d(points, centroid, direction, colors=None):
    """Show point cloud and a red fitted line in Open3D (segfault-safe)."""
    # Safety: drop NaN/Inf from BOTH points AND colors
    mask = np.all(np.isfinite(points), axis=1)
    print("created masks")
    points = points[mask]
    if colors is not None:
        colors = colors[mask]  # ✅ CRITICAL FIX: filter colors too!
        # Also check for NaN/Inf in colors themselves
        color_mask = np.all(np.isfinite(colors), axis=1)
        if not np.all(color_mask):
            print(f"Warning: Found {np.sum(~color_mask)} invalid colors, filtering...")
            points = points[color_mask]
            colors = colors[color_mask]
    print("filtered points and colors")
    
    if points.size == 0:
        print("No valid points to visualize.")
        return

    cv2.destroyAllWindows()  # ✅ prevent GUI thread conflict
    cv2.waitKey(1)  # Give time for windows to close
    print("destroyed cv2 windows")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None and len(colors) == len(points):
        print(f"adding colors: {colors.shape}, dtype: {colors.dtype}")
        # Ensure colors are float64 and in valid range [0, 1]
        colors = np.asarray(colors, dtype=np.float64)
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        print("using uniform color")
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # Red line along the fitted direction - validate centroid and direction
    print(f"centroid: {centroid}, direction: {direction}")
    if not (np.all(np.isfinite(centroid)) and np.all(np.isfinite(direction))):
        print("Warning: centroid or direction contains NaN/Inf, recomputing from filtered points")
        centroid = points.mean(axis=0)
        pts_centered = points - centroid
        U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
        direction = Vt[0]
    
    # Make line much longer - scale based on point cloud extent
    cloud_extent = np.max(np.linalg.norm(points - centroid, axis=1))
    t = max(cloud_extent * 2.0, 5.0)  # At least 2x the cloud extent or 5 units
    print(f"Cloud extent: {cloud_extent}, line extension: {t}")
    
    line_pts = np.array([centroid - direction * t, centroid + direction * t], dtype=np.float64)
    print(f"line_pts: {line_pts}")
    
    if not np.all(np.isfinite(line_pts)):
        print("Error: line_pts contains NaN/Inf, skipping line visualization")
        o3d.visualization.draw_geometries([pcd], window_name="3D Fit Visualization", width=960, height=720)
        return
    
    # Create LineSet properly with explicit conversions
    line_set = o3d.geometry.LineSet()
    # Ensure line_pts is contiguous and correct shape
    line_pts_clean = np.ascontiguousarray(line_pts, dtype=np.float64)
    line_set.points = o3d.utility.Vector3dVector(line_pts_clean)
    # Ensure lines array is correct type
    lines_array = np.array([[0, 1]], dtype=np.int32)
    line_set.lines = o3d.utility.Vector2iVector(lines_array)
    # Ensure colors array is correct type and shape
    colors_array = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    line_set.colors = o3d.utility.Vector3dVector(colors_array)
    print("created line set")

    o3d.visualization.draw_geometries(
        [pcd, line_set],
        window_name="3D Fit Visualization",
        width=960,
        height=720,
    )

def main():
    print("Starting ZED capture + segmentation tool")
    saved = capture_frame_and_save()
    if saved is None:
        print("No capture saved. Exiting.")
        return

    rgb_path, depth_path, intr_path = saved
    img = cv2.imread(rgb_path)
    depth_arr = np.load(depth_path)
    intr = np.load(intr_path)

    seg = PolySegmenter(img, depth_arr, intr)
    seg.run()

if __name__ == '__main__':
    main()