import numpy as np
import cv2
import open3d as o3d
import os
from line_segmenter_v2 import make_pcd
CAPTURE_DIR = "zed_captures"


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
        # o3d.visualization.draw_geometries([pcd_special], window_name="special Point Cloud")
    else:
        pcd_special = o3d.geometry.PointCloud()

    # ---------------- Red fitted line ----------------
    # Ensure centroid and direction are 1D arrays of length 3
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    cloud_extent = np.max(np.linalg.norm(points_orig - centroid, axis=1))
    t = max(cloud_extent * 2.0, 5.0)
    line_start = centroid - direction * t
    line_end   = centroid + direction * t
    line_pts = np.vstack([line_start, line_end])  # shape (2,3)
    line_set = make_lineset_safe(line_pts, color=[1.0, 0.0, 0.0])

    # ---------------- Visualize ----------------
    o3d.visualization.draw_geometries([pcd_full, pcd_fitted], window_name="All Point Cloud")


# ---------- SEGMENTATION ----------
class SimpleBrushSegmenter:
    def __init__(self, image, depth, intr):
        self.image = image.copy()
        self.clone = image.copy()
        self.depth = depth
        self.intr = intr
        self.brush_radius = 3  # smaller brush
        self.mask = np.zeros(image.shape[:2], np.uint8)
        self.points = []
        self.special_points = []

        self.window = "Segmentation"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse_cb)

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
        return self.mask


def mask_to_points(mask, depth, intr, rgb):
    fx, fy, cx, cy = intr
    ys, xs = np.where(mask == 255)
    zs = depth[ys, xs]
    valid = np.isfinite(zs) & (zs > 0)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    if len(xs) == 0:
        return None, None

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    pts = np.vstack((X, Y, Z)).T
    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[ys, xs] / 255.0
    return pts, colors

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

# ---------- MAIN ----------
def main():
    # Load data
    timestamp = "1759846948"  # Replace with your actual timestamp
    rgb_path = os.path.join(CAPTURE_DIR, f"rgb_{timestamp}.png")
    depth_path = os.path.join(CAPTURE_DIR, f"depth_{timestamp}.npy")
    intr_path = os.path.join(CAPTURE_DIR, f"intrinsics_{timestamp}.npy")

    img = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    intr = np.load(intr_path)
    
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

    # Extract original points/colors
    points_orig, colors_orig = rgbd_to_points(img, depth, intr)

    seg = SimpleBrushSegmenter(img, depth, intr)
    mask = seg.run()

    # Convert mask → 3D points
    pts, colors = mask_to_points(mask, depth, intr, img)
    if pts is None:
        print("No valid 3D points found.")
        return

    centroid, direction = fit_line_pca(pts)

    # After segmentation, allow user to select 2 points
    special_points = select_two_points(img)
    print(special_points)
    if len(special_points) != 2:
        print("Need exactly 2 points.")
        return

    # Backproject 2D → 3D
    fx, fy, cx, cy = intr
    special_pts_3d = []
    # Project onto fitted line
    projected = np.array([project_pixel_on_line(p, intr, centroid, direction) for p in special_points])
    print("Projected points on line:", projected)

    for i, (x, y) in enumerate(special_points):
        z = depth[int(y), int(x)]
        if np.isnan(z) or z <= 0:
            point_3d = projected[i]
        else:
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            point_3d = np.array([X, Y, z])
        special_pts_3d.append(point_3d)
    special_pts_3d = np.array(special_pts_3d)

    
   
    # Compute distance in mm
    dist_mm = np.linalg.norm(projected[0] - projected[1]) * 1000
    print(f"Distance between the two special points (projected): {dist_mm:.2f} mm")

    # Find lowest segmented point (in image coords)
    lowest_idx = np.argmax([p[1] for p in seg.points])
    lowest_point = seg.points[lowest_idx]
    print(f"Lowest segmented point (image coords): {lowest_point}")
    cv2.destroyAllWindows()

    print_nan_inf("points_orig", points_orig)
    print_nan_inf("colors_orig", colors_orig)
    print_nan_inf("centroid", centroid)
    print_nan_inf("direction", direction)
    print_nan_inf("pts", pts)
    print_nan_inf("projected", projected)


    visualize_all(points_orig, colors_orig, centroid, direction, pts, projected)



 
if __name__ == "__main__":
    main()
