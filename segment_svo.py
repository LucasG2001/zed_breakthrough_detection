import numpy as np
import cv2
import open3d as o3d
import os
from line_segmenter_v2 import make_pcd, angle_between_vectors
from constrained_fit import fit_3d_line_with_pixel_constraint_v3
from load_svos import  list_svos, load_from_svo
import json
import datetime

CAPTURE_DIR = "zed_captures"

def project_3d_to_pixel(pt3d, fx, fy, cx, cy):
        if pt3d[2] == 0:
            return None
        u = int(round(pt3d[0] * fx / pt3d[2] + cx))
        v = int(round(pt3d[1] * fy / pt3d[2] + cy))
        return (u, v)
         

# ---------- FIT FUNCTIONS ----------
def fit_line_pca(points):
    centroid = points.mean(axis=0)
    pts_centered = points - centroid
    U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    direction = Vt[0]
    return centroid, direction

def fit_line_pca_weighted(points, confidence_map):
    """
    Fits a line to 2D or 3D points using weighted PCA.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, D) where D is 2 or 3.
    confidence_map : np.ndarray
        Array of shape (N,) with values 0–100 (0=good, 100=bad).
    
    Returns
    -------
    centroid : np.ndarray
        Weighted centroid of the points.
    direction : np.ndarray
        Principal direction (unit vector) of the best-fit line.
    """
    # Convert confidence to weights: 0 is bad, 100 is good
    weights = 100.0 - confidence_map
    weights = np.clip(weights, 0, None)  # no negative weights
    weights = weights / np.sum(weights)  # normalize to sum=1

    # Weighted centroid
    centroid = np.average(points, axis=0, weights=weights)

    # Center points by weighted centroid
    pts_centered = points - centroid

    # Weighted covariance matrix
    cov = (pts_centered * weights[:, np.newaxis]).T @ pts_centered

    # Eigen decomposition (equivalent to PCA)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]

    return centroid, direction


def make_lineset_safe(line_pts, color=[1, 0, 0]):
    if line_pts is None or len(line_pts) < 2:
        print("⚠️ make_lineset_safe(): invalid line points")
        return o3d.geometry.LineSet()

    line_pts = np.ascontiguousarray(line_pts, dtype=np.float64)
    lines = np.array([[0, 1]], dtype=np.int32)
    colors = np.ascontiguousarray(np.array([color], dtype=np.float64))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(line_pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def rgbd_to_points(rgb, depth, intrinsics, z_threshold):
    """
    Convert full RGB-D image to 3D point cloud and corresponding colors.
    
    Args:
        rgb: (H, W, 3) RGB image, dtype uint8
        depth: (H, W) depth map in meters
        intrinsics: (fx, fy, cx, cy)
    Returns:
        pts: (N, 3) array of valid 3D points
        colors: (N, 3) array of corresponding RGB colors in [0, 1]
    """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth

    # ✅ Valid depth mask: positive, finite, and not NaN
    valid_mask = (z > 0) & np.isfinite(z) & (z < z_threshold)

    # Compute 3D coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)

    # Apply mask
    pts = pts[valid_mask]
    colors = rgb[valid_mask] / 255.0

    return pts, colors




def project_point_on_line(point, line_origin, line_dir):
    """Project a point onto a 3D line."""
    v = point - line_origin
    t = np.dot(v, line_dir)
    return line_origin + t * line_dir

def project_pixel_on_line(p, intrinsics, line_origin, line_dir, camera_origin=np.array([0,0,0])):
    """
    Project a pixel onto a 3D line using camera intrinsics, even if depth is unknown.
    
    Args:
        u, v: Pixel coordinates
        intrinsics: 3x3 camera intrinsic matrix
        line_origin: 3D point on the line
        line_dir: Unit direction vector of the line
        camera_origin: 3D camera center (default [0,0,0])
    
    Returns:
        3D point on the line corresponding to the pixel
    """

    u,v = p
    fx, fy, cx, cy = intrinsics

    # 1. Compute the ray from camera through the pixel
    ray_dir = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    ray_dir /= np.linalg.norm(ray_dir)

    # 2. Solve for closest point between line and ray
    p1, d1 = camera_origin, ray_dir
    p2, d2 = line_origin, line_dir / np.linalg.norm(line_dir)

    # Compute parameters t1, t2 for closest points on the two lines
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a*c - b*b
    if np.abs(denom) < 1e-6:
        # Lines are nearly parallel; fallback to line origin
        return line_origin

    t2 = (b*d - a*e) / denom
    # t1 = (b*t2 - d) / a  # optional if you want closest ray point

    # 3. Compute the closest point on the line
    closest_point = p2 + t2 * d2
    return closest_point

# ---------- VISUALIZATION ----------
def visualize_all(points_orig, colors_orig, centroids, directions, fitted_points_after,
                  special_points_projected):
    """
    Robust visualization:
      - Full original point cloud (RGB)
      - Fitted points (green, fat)
      - Special points (yellow, very fat)
      - Fitted line (red)
    """
    for (centroid, direction) in zip(centroids, directions):    
        if not np.all(np.isfinite(centroid)) or not np.all(np.isfinite(direction)):
            print("Invalid centroid or direction for line!")
            return

    # ---------------- Full RGB cloud ----------------
    print("making full pcd")
    pcd_full = make_pcd(points_orig, colors_orig)
    # o3d.visualization.draw_geometries([pcd_full], window_name="Full Point Cloud")

    # ---------------- Green fitted points ----------------
    if len(fitted_points_after) > 0:
        green_colors = np.tile(np.array([0.0, 1.0, 0.0]), (len(fitted_points_after), 1))
        print("making fitted pcd")
        pcd_fitted = make_pcd(fitted_points_after, green_colors)
        # o3d.visualization.draw_geometries([pcd_fitted], window_name="Fitted Point Cloud")
    else:
        pcd_fitted = o3d.geometry.PointCloud()

    # ---------------- Special points (yellow) ----------------
    if len(special_points_projected) > 0:
        yellow_colors = np.tile(np.array([1.0, 1.0, 0.0]), (len(special_points_projected), 1))
        print("making special pcd")
        pcd_special = make_pcd(special_points_projected, yellow_colors)
        #o3d.visualization.draw_geometries([pcd_special], window_name="special Point Cloud")
    else:
        pcd_special = o3d.geometry.PointCloud()

    # ---------------- Red fitted line ----------------
    line_sets = []
    # RGB -> red green blue color
    colors = [np.array([[1.0, 0.0, 0.0]], dtype=np.float64), np.array([[0.0, 1.0, 0.0]], dtype=np.float64), np.array([[0.0, 0.0, 1.0]], dtype=np.float64)]
    for i, (centroid, direction) in enumerate(zip(centroids, directions)):
    # Ensure centroid and direction are 1D arrays of length 3
        centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        direction = direction / np.linalg.norm(direction)
        cloud_extent = np.max(np.linalg.norm(fitted_points_after - centroid, axis=1))
        t = min(max(cloud_extent * 2.0, 0.1), 0.6)  # between 0.1 and 1.0 meters
        print("Line endpoints:", centroid - direction * t, centroid + direction * t)
        line_pts = np.array([centroid - direction * t, centroid + direction * t], dtype=np.float64)
        line_set = o3d.geometry.LineSet()
        line_pts_clean = np.ascontiguousarray(line_pts, dtype=np.float64)
        line_set.points = o3d.utility.Vector3dVector(line_pts_clean)

        lines_array = np.array([[0, 1]], dtype=np.int32)
        line_set.lines = o3d.utility.Vector2iVector(lines_array)
        colors_array = colors[i]
        line_set.colors = o3d.utility.Vector3dVector(colors_array)
        line_sets.append(line_set)

    print("Created line set — ready to visualize")

    # ---------------- Visualize ----------------
    o3d.visualization.draw_geometries([pcd_full, pcd_fitted], window_name="All Point Cloud without fitted lines")
    o3d.visualization.draw_geometries([pcd_full, pcd_fitted] + line_sets[1:3], window_name="All Point Cloud with fitted lines PCA  gb =/weighted/pca")
    o3d.visualization.draw_geometries([pcd_full, pcd_fitted] + line_sets, window_name="All Point Cloud with all fitted lines RGB = constr/weighted/pca")


# ---------- SEGMENTATION ----------
class SimpleBrushSegmenter:
    def __init__(self, image, depth, intr, conf=None):
        self.image = image.copy()
        self.clone = image.copy()
        self.depth = depth
        self.intr = intr
        self.conf = conf
        self.brush_radius = 6  # smaller brush
        self.mask = np.zeros(image.shape[:2], np.uint8)
        self.points = []
        self.special_points = []

        self.window = "Segmentation"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse_cb)

        # --- Display confidence if provided ---
        if self.conf is not None:
            conf_disp = 100 - self.conf
            conf_disp = np.clip(conf_disp, 0, 100)
            conf_norm = (conf_disp / 100.0 * 255).astype(np.uint8)
            cv2.imshow("Confidence (100-confidence, grayscale)", conf_norm)
            cv2.waitKey(1)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(self.clone, (x, y), self.brush_radius, (0, 255, 0), -1)
            cv2.circle(self.mask, (x, y), self.brush_radius, 255, -1)
            self.points.append((x, y))

    def run(self):
        while True:
            cv2.imshow(self.window, self.clone)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('f'):
                print("Segmentation finalized.")
                break
            elif key == ord('r'):
                self.clone = self.image.copy()
                self.mask[:] = 0
                self.points = []

        cv2.destroyWindow(self.window)
        if self.conf is not None:
            cv2.destroyWindow("Confidence (100-confidence, grayscale)")
        return self.mask


def mask_to_points(mask, depth, intr, rgb, confidence_map, z_threshold=0.2):
    fx, fy, cx, cy = intr
    ys, xs = np.where(mask == 255)
    zs = depth[ys, xs]
    confidence_array = confidence_map[ys, xs]
    valid = np.isfinite(zs) & (zs > 0) & (zs < z_threshold) # filter points that are too far away
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    if len(xs) == 0:
        return None, None

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    confidences = confidence_array[valid]
    pts = np.vstack((X, Y, Z)).T
    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[ys, xs] / 255.0
    return pts, colors, confidences

# ---------- SECOND STAGE: SELECT 2 POINTS ----------
def select_points(image, n_points=2, window_args="(selection of start and end points of drill bit)"):
    """Interactive 2-point selector with visual feedback."""
    temp = image.copy()
    window = f"Select {n_points} Points" + " " + window_args
    points = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal points, temp
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < n_points:
                points.append((x, y))
                cv2.circle(temp, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow(window, temp)
            if len(points) == n_points:
                print(f"✅ {n_points} points selected. Press 'f' to finalize.")

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_cb)

    print(f"🖱️ Click {n_points} points. Press 'f' when done or 'r' to reset.")
    while True:
        cv2.imshow(window, temp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            temp = image.copy()
            points = []
            print("↩️  Reset points.")
        elif key == ord('f') and len(points) == n_points:
            break
        elif key == ord('q'):
            points = []
            break

    cv2.destroyWindow(window)
    return points

def show_depth_image(depth):
    # --- Display grayscale depth image ---
     depth_disp = np.copy(depth)
     depth_disp[~np.isfinite(depth_disp)] = 0
     valid = depth_disp > 0
     if np.any(valid):
         min_z = np.percentile(depth_disp[valid], 2)
         max_z = np.percentile(depth_disp[valid], 98)
     else:
         min_z, max_z = 0, 1
     depth_norm = np.clip((depth_disp - min_z) / (max_z - min_z + 1e-6), 0, 1)
     depth_vis = (depth_norm * 255).astype(np.uint8)
     cv2.imshow("Depth Grayscale", depth_vis)
     cv2.waitKey(1)

# ---------- MAIN ----------
def compute_overshoot(is_manual: bool, file_path, z_threshold=0.25):
    # Load data
    if is_manual:
        img, depth, intr, conf, gravity_dir = load_from_svo(file_path)
        file = os.path.basename(file_path)
        reference_angle = float(file.split("_")[2])
    else:
         # Load data
        files = [f for f in os.listdir(file_path)]
        print(files)
        for f in files:
            if f.endswith("confidence.npy"):
                conf_path = os.path.join(file_path, f) 
            elif f.endswith("depth.npy"):
                depth_path = os.path.join(file_path, f) 
            elif f.endswith("intrinsics.npy"):
                intr_path = os.path.join(file_path, f) 
            elif f.endswith("rgb.png"):
                rgb_path = os.path.join(file_path, f) 
                reference_angle = float(f.split("_")[2])
            elif f.endswith("gravity.npy"):
                gravity_path = os.path.join(file_path, f)

        img = cv2.imread(rgb_path)
        depth = np.load(depth_path)
        intr = np.load(intr_path)
        conf = np.load(conf_path)
        gravity_dir = np.load(gravity_path)
        print(f"gravity direction (90 deg) is {gravity_dir}")
        print(f"reference_direction is {reference_angle}")

    # Extract original points/colors
    fx, fy, cx, cy = intr
    points_orig, colors_orig = rgbd_to_points(img, depth, intr, z_threshold)
    
    # Pass confidence to segmenter
    seg = SimpleBrushSegmenter(img, depth, intr, conf=conf)
    mask = seg.run()
    


    # After segmentation, allow user to select 2 points
    special_points = select_points(img, 2, window_args="(select start and end points of drill bit)")
    if len(special_points) != 2:
        print("Need exactly 2 points.")
        return
    
    exit_point = select_points(img, 1, window_args="(select exit point of drill bit outside the bone)")
    if len(exit_point) != 1:
        print("Need exactly 1 points.")
        return


    # Convert mask → 3D points
    pts, colors, confidences = mask_to_points(mask, depth, intr, img, conf, z_threshold)
    pts, colors, confidences = pts[confidences < 95], colors[confidences < 95] , confidences[confidences < 95]   # filter out low-confidence points
    print(f"number of valid points is {len(pts)}")
    if pts is None:
        print("No valid 3D points found.")
        return

    centroid_pca, direction_pca = fit_line_pca(pts)
    centroid_weighted, direction_weighted = fit_line_pca_weighted(pts, confidences)
    constrained_line = fit_3d_line_with_pixel_constraint_v3(pts, centroid_weighted, direction_weighted, confidences, special_points[0], special_points[1],
                                                            fx, fy, cx, cy, target_distance_mm=122.0, maxiter=7000, verbose = True)
    
    centroid_constrained = constrained_line['p0']
    direction_constrained = constrained_line['v']

    # Backproject 2D → 3D
    # Project onto fitted line
    projectd_pca = np.array([project_pixel_on_line(p, intr, centroid_pca, direction_pca) for p in special_points])
    projected_weighted = np.array([project_pixel_on_line(p, intr, centroid_weighted, direction_weighted) for p in special_points])
    projected_constrained = np.array([project_pixel_on_line(p, intr, centroid_constrained, direction_constrained) for p in special_points])
    #exit points
    projected_exit_point_constrained = np.array([project_pixel_on_line(p, intr, centroid_constrained, direction_constrained) for p in exit_point])
    projected_exit_point_pca = np.array([project_pixel_on_line(p, intr, centroid_pca, direction_pca) for p in exit_point])
    projected_exit_point_weighted = np.array([project_pixel_on_line(p, intr, centroid_weighted, direction_weighted) for p in exit_point])

    drill_tip_contstrained = projected_constrained[1 if special_points[0][1] < special_points[1][1] else 0]
    drill_tip_pca = projectd_pca[1 if special_points[0][1] < special_points[1][1] else 0]
    drill_tip_weighted = projected_weighted[1 if special_points[0][1] < special_points[1][1] else 0]

   
    # Compute distance in mm
    dist_pca = np.linalg.norm(projectd_pca[0] - projectd_pca[1]) * 1000
    # print(f"Distance between the two special points (PCA): {dist_pca:.2f} mm")
    dist_weighted = np.linalg.norm(projected_weighted[0] - projected_weighted[1]) * 1000
    # print(f"Distance between the two special points (Weighted PCA): {dist_weighted:.2f} mm")
    dist_mm = np.linalg.norm(projected_constrained[0] - projected_constrained[1]) * 1000
    # print(f"Distance between the two special points (projected): {dist_mm:.2f} mm")
    soft_tissue_penetration_mm_constrained = np.linalg.norm(drill_tip_contstrained - projected_exit_point_constrained) * 1000
    soft_tissue_penetration_mm_weighted = np.linalg.norm(drill_tip_weighted - projected_exit_point_weighted) * 1000
    soft_tissue_penetration_mm_pca = np.linalg.norm(drill_tip_pca - projected_exit_point_pca) * 1000

    centroids = [centroid_constrained, centroid_weighted, centroid_pca]
    directions = [direction_constrained, direction_weighted, direction_pca]
    orientation_errors = []
    # compute orientation errors
    for drill_direction in directions:
        print("Line direction:", drill_direction)
        print("gravity direction:", gravity_dir)
        reference_deviation = 90.0 - reference_angle
        error_to_gravity_dir = angle_between_vectors(drill_direction, gravity_dir)
        print(f"error to gravity is {error_to_gravity_dir}")
        orientation_errors.append(np.abs(reference_deviation-error_to_gravity_dir)) # compute error via desired deviation - total dev

    # visualize_all(points_orig, colors_orig, centroids, directions, pts, projected_constrained)

    return {"distance_pca_mm": dist_pca,
            "distance_weighted_mm": dist_weighted,
            "distance_constrained_mm": dist_mm,
            "orientation_error_constrained": orientation_errors[0],
            "orientation_error_weighted": orientation_errors[1],
            "orientation_error_pca": orientation_errors[2],
            "soft_tissue_penetration_mm_constrained": soft_tissue_penetration_mm_constrained,
            "soft_tissue_penetration_mm_weighted": soft_tissue_penetration_mm_weighted,
            "soft_tissue_penetration_mm_pca": soft_tissue_penetration_mm_pca
            }, mask
    


if __name__ == "__main__":
    # (TODO: erode edges of mask)
    # MANUAL or ROBOTIC

    manual = False
    # -----------------------------
    # Parameters
    # -----------------------------
    PARTICIPANT = 21
    RUN_NUMBER = 3
    
    dicts = [] # will be filled
    # Get current timestamp Format: "DD-MM-YYYY_HHMMSS"
    timestamp_str = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    
    # -----------4.38------------------
    # Manual Drilling
    # -----------------------------
    if manual:
        results_dir = os.path.join("results_manual", str(PARTICIPANT))
        DATA_DIR = f"manual_data/{PARTICIPANT}_data_manual/{PARTICIPANT}"
        svo_path = list_svos(DATA_DIR)[RUN_NUMBER-1]
        print(f"--- Processing run {RUN_NUMBER} ---")
        out, mask = compute_overshoot(is_manual=True, file_path=DATA_DIR + "/" + svo_path)
    # -----------------------------
    # Robotic Drilling
    # -----------------------------
    else:
        results_dir = os.path.join("results_robotic", str(PARTICIPANT))
        DATA_DIR = "saved_robot_data"
        participant_str = f"{PARTICIPANT}_data_robotic"
        run_str = f"run{RUN_NUMBER}"
        data_directory_full = os.path.join(DATA_DIR, participant_str, run_str)
        print(f"--- Processing run {RUN_NUMBER} ---")
        out, mask = compute_overshoot(is_manual=False, file_path=data_directory_full)
    if out is not None:
        # Assign to the output dictionary
        out['timestamp'] = timestamp_str
        dicts.append(out)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{RUN_NUMBER}.json")
    output_mask = os.path.join(results_dir, f"{RUN_NUMBER}_mask.npy")
    # -----------------------------
    # Format table nicely for display
    # -----------------------------
    for d in dicts:
        print("-" * 70)
        print(f"{'Metric':30} | {'PCA':>10} | {'Weighted':>14} | {'Constrained':>17}")
        print("-" * 70)
        print(f"{'control_distance':30} | {d['distance_pca_mm']:10.2f} | {d['distance_weighted_mm']:14.2f} | {d['distance_constrained_mm']:17.2f}")
        print(f"{'soft_tissue_penetration':30} | {d['soft_tissue_penetration_mm_pca']:10.2f} | {d['soft_tissue_penetration_mm_weighted']:14.2f} | {d['soft_tissue_penetration_mm_constrained']:17.2f}")
        print(f"{'orientation_error':30} | {d['orientation_error_pca']:10.2f} | {d['orientation_error_weighted']:14.2f} | {d['orientation_error_constrained']:17.2f}")
        print("-" * 70)

    # -----------------------------
    # Save to JSON
    # -----------------------------
    with open(output_file, 'w') as f:
        json.dump(dicts, f, indent=4)
        np.save(output_mask, mask)

    print(f"Saved results to {output_file}")
