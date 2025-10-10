import numpy as np
import cv2
import open3d as o3d
import os
from line_segmenter_v2 import make_pcd
from constrained_fit import fit_3d_line_with_pixel_constraint_v2
CAPTURE_DIR = "zed_captures"

def project_3d_to_pixel(pt3d, fx, fy, cx, cy):
        if pt3d[2] == 0:
            return None
        u = int(round(pt3d[0] * fx / pt3d[2] + cx))
        v = int(round(pt3d[1] * fy / pt3d[2] + cy))
        return (u, v)


# ---- Print NaN and Inf diagnostics ----
def print_nan_inf(name, arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"{name}: empty")
        return
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    if np.any(nan_mask):
        print(f"{name}: NaN at indices {np.argwhere(nan_mask)}")
    if np.any(inf_mask):
        print(f"{name}: Inf at indices {np.argwhere(inf_mask)}")
    if not np.any(nan_mask) and not np.any(inf_mask):
        print(f"{name}: OK (no NaN or Inf)")
            

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


def rgbd_to_points(rgb, depth, intrinsics):
    """
    Convert full RGB-D image to 3D points and colors.
    """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics

    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid_mask = z > 0

    # Compute 3D coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)

    # Filter out invalid points
    pts = pts[valid_mask]

    # Get corresponding colors (scale to 0-1)
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
def visualize_all(points_orig, colors_orig, centroid, direction, fitted_points_after,
                  special_points_projected):
    """
    Robust visualization:
      - Full original point cloud (RGB)
      - Fitted points (green, fat)
      - Special points (yellow, very fat)
      - Fitted line (red)
    """
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
    # Ensure centroid and direction are 1D arrays of length 3
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    direction = direction / np.linalg.norm(direction)
    cloud_extent = np.max(np.linalg.norm(points_orig - centroid, axis=1))
    t = min(max(cloud_extent * 2.0, 0.1), 1.0)  # between 0.1 and 1.0 meters
    print("Line endpoints:", centroid - direction * t, centroid + direction * t)
    line_pts = np.array([centroid - direction * t, centroid + direction * t], dtype=np.float64)
    line_set = o3d.geometry.LineSet()
    line_pts_clean = np.ascontiguousarray(line_pts, dtype=np.float64)
    line_set.points = o3d.utility.Vector3dVector(line_pts_clean)

    lines_array = np.array([[0, 1]], dtype=np.int32)
    line_set.lines = o3d.utility.Vector2iVector(lines_array)

    colors_array = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    line_set.colors = o3d.utility.Vector3dVector(colors_array)

    print("Created line set — ready to visualize")

    # ---------------- Visualize ----------------
    o3d.visualization.draw_geometries([pcd_full, pcd_fitted, line_set, pcd_special], window_name="All Point Cloud")


# ---------- SEGMENTATION ----------
class SimpleBrushSegmenter:
    def __init__(self, image, depth, intr, conf=None):
        self.image = image.copy()
        self.clone = image.copy()
        self.depth = depth
        self.intr = intr
        self.conf = conf
        self.brush_radius = 3  # smaller brush
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


def mask_to_points(mask, depth, intr, rgb, confidence_map):
    fx, fy, cx, cy = intr
    ys, xs = np.where(mask == 255)
    zs = depth[ys, xs]
    confidence_array = confidence_map[ys, xs]
    valid = np.isfinite(zs) & (zs > 0)
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
def select_two_points(image):
    """Interactive 2-point selector with visual feedback."""
    temp = image.copy()
    window = "Select Two Points"
    points = []

    def mouse_cb(event, x, y, flags, param):
        nonlocal points, temp
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                cv2.circle(temp, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow(window, temp)
            if len(points) == 2:
                print("✅ 2 points selected. Press 'f' to finalize.")

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_cb)

    print("🖱️ Click two points. Press 'f' when done or 'r' to reset.")
    while True:
        cv2.imshow(window, temp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            temp = image.copy()
            points = []
            print("↩️  Reset points.")
        elif key == ord('f') and len(points) == 2:
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
def compute_overshoot(timestamp):
    # Load data
    rgb_path = os.path.join(CAPTURE_DIR, f"rgb_{timestamp}.png")
    depth_path = os.path.join(CAPTURE_DIR, f"depth_{timestamp}.npy")
    intr_path = os.path.join(CAPTURE_DIR, f"intrinsics_{timestamp}.npy")
    conf_path = os.path.join(CAPTURE_DIR, f"confidence_{timestamp}.npy")

    img = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    intr = np.load(intr_path)
    conf = np.load(conf_path)
    
    # show_depth_image(depth)

    # Extract original points/colorssss
    fx, fy, cx, cy = intr
    points_orig, colors_orig = rgbd_to_points(img, depth, intr)

    # Pass confidence to segmenter
    seg = SimpleBrushSegmenter(img, depth, intr, conf=conf)
    mask = seg.run()

    # After segmentation, allow user to select 2 points
    special_points = select_two_points(img)
    print(special_points)
    if len(special_points) != 2:
        print("Need exactly 2 points.")
        return


    # Convert mask → 3D points
    pts, colors, confidences = mask_to_points(mask, depth, intr, img, conf)
    if pts is None:
        print("No valid 3D points found.")
        return

    centroid_pca, direction_pca = fit_line_pca(pts)
    centroid_weighted, direction_weighted = fit_line_pca_weighted(pts, confidences)
   

    # Backproject 2D → 3D
    special_pts_3d = []
    # Project onto fitted line
    projectd_pca = np.array([project_pixel_on_line(p, intr, centroid_pca, direction_pca) for p in special_points])
    projected_weighted = np.array([project_pixel_on_line(p, intr, centroid_weighted, direction_weighted) for p in special_points])

    constrained_line = fit_3d_line_with_pixel_constraint_v2(pts, confidences, special_points[0], special_points[1],
                                            fx, fy, cx, cy,
                                            target_distance_mm=145.0, maxiter=10000)
    
    print(constrained_line)
    
    centroid = constrained_line['p0']
    direction = constrained_line['v']
    
    projected_constrained = np.array([project_pixel_on_line(p, intr, centroid, direction) for p in special_points])
    for i, (x, y) in enumerate(special_points):
            point_3d = projected_constrained[i]
            special_pts_3d.append(point_3d)
    special_pts_3d = np.array(special_pts_3d)

   
    # Compute distance in mm
    dist_pca = np.linalg.norm(projectd_pca[0] - projectd_pca[1]) * 1000
    # print(f"Distance between the two special points (PCA): {dist_pca:.2f} mm")
    dist_weighted = np.linalg.norm(projected_weighted[0] - projected_weighted[1]) * 1000
    # print(f"Distance between the two special points (Weighted PCA): {dist_weighted:.2f} mm")
    dist_mm = np.linalg.norm(projected_constrained[0] - projected_constrained[1]) * 1000
    # print(f"Distance between the two special points (projected): {dist_mm:.2f} mm")


    # visualize_all(points_orig, colors_orig, centroid, direction, pts, projected_constrained)

    return {"distance_pca_mm": dist_pca,
            "distance_weighted_mm": dist_weighted,
            "distance_constrained_mm": dist_mm,}
    


if __name__ == "__main__":
    # TODO: filter pointclouds
    # TODO: visualize different lines
    # TODO: erode edges of mask
    # TODO: segment third points
    
    timestamps = ["1760020410"]  # Replace with your actual timestamp
    #compute_overshoot(timestamp)
    # timestamps = ["1760017023", "1760020318", "1760020342", "1760020362", "1760020385", "1760020410"]
    dicts = []
    for ts in timestamps:
        print(f"--- Processing timestamp {ts} ---")
        out = compute_overshoot(ts)
        if out is not None:
            out['timestamp'] = ts
            dicts.append(out)

     # Print results in tabular form
    if dicts:
        print("\nResults:")
        print(f"{'Timestamp':>12} | {'PCA (mm)':>10} | {'Weighted (mm)':>14} | {'Constrained (mm)':>17}")
        print("-" * 60)
        for d in dicts:
            print(f"{d['timestamp']:>12} | {d['distance_pca_mm']:10.2f} | {d['distance_weighted_mm']:14.2f} | {d['distance_constrained_mm']:17.2f}")

