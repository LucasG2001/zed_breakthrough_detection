"""
zed_line_fit_reconstruct_safe.py

Robust segmentation -> PCA & RANSAC line fit -> depth reconstruction -> visualization.

Requirements:
  - numpy, opencv-python, open3d, scikit-learn

Behavior:
  - Load RGB, Depth (meters), Intrinsics (fx,fy,cx,cy) from CAPTURE_DIR
  - User paints a mask (brush ~0.6x). Press 'f' to finalize, 'r' to reset, 'q' to quit.
  - Valid depths -> fit PCA line and RANSAC line.
  - Show full colored point cloud, PCA line (red), RANSAC line (blue), segmented line points (green).
  - Reconstruct missing depths (masked invalid pixels) by projecting their camera rays onto the RANSAC line.
  - Show reconstructed full cloud (original colors) and the line points in green.
  - Print 3D coordinate of the lowest segmented pixel (after reconstruction).
"""

import os
import cv2
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RANSACRegressor

CAPTURE_DIR = "zed_captures"

# ---------- Helpers ----------

def ensure_contiguous_f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)

def ensure_contiguous_i32(a):
    return np.ascontiguousarray(a, dtype=np.int32)

def safe_show_open3d(geoms, window_name="Open3D", width=1280, height=720):
    """Create an Open3D Visualizer window and show geometries robustly."""
    try:
        # Close OpenCV windows first (very important to avoid GUI conflicts)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=width, height=height)
        for g in geoms:
            vis.add_geometry(g)
        vis.get_render_option().point_size = 2.0
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print("Open3D visualization failed:", e)


def full_pointcloud_from_depth(depth_arr, intr, rgb_img=None):
    """
    Build full point cloud (only for valid depths).
    Returns: points (N,3), colors (N,3), ys, xs, idx_map (H,W -> -1 or index)
    """
    H, W = depth_arr.shape
    fx, fy, cx, cy = intr

    valid_mask = np.isfinite(depth_arr) & (depth_arr > 0.01)
    ys, xs = np.where(valid_mask)
    zs = depth_arr[ys, xs]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    pts = np.vstack((X, Y, Z)).T.astype(np.float64)

    colors = None
    if rgb_img is not None:
        rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        sampled = rgb[ys, xs].astype(np.float64) / 255.0
        colors = sampled

    # index mapping
    idx_map = -np.ones_like(depth_arr, dtype=np.int32)
    idx_map[ys, xs] = np.arange(len(ys), dtype=np.int32)

    return pts, colors, ys, xs, idx_map


# ---------- Segmentation UI ----------

class BrushSegmenter:
    def __init__(self, image, depth_arr, intr, brush_radius=6):
        self.image = image.copy()
        self.clone = image.copy()
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.brush_radius = max(1, int(0.6 * brush_radius))  # scale by 0.6
        self.window = "Brush: hold left mouse, 'f' finalize, 'r' reset, 'q' quit"
        self.depth_arr = depth_arr
        self.intr = intr
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)
        self.drawing = False

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.mask, (x, y), self.brush_radius, 255, -1)
            cv2.circle(self.clone, (x, y), self.brush_radius, (0, 255, 0), -1)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(self.mask, (x, y), self.brush_radius, 255, -1)
            cv2.circle(self.clone, (x, y), self.brush_radius, (0, 255, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def run(self):
        while True:
            cv2.imshow(self.window, self.clone)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('r'):
                self.mask[:] = 0
                self.clone = self.image.copy()
                print("Mask reset.")
            elif k == ord('f'):
                cv2.destroyAllWindows()
                return self.mask
            elif k == ord('q'):
                cv2.destroyAllWindows()
                return None


# ---------- Line fitting ----------

def fit_line_pca(points):
    """
    PCA line fit: returns centroid (point on line) and unit direction vector.
    points: (N,3)
    """
    centroid = points.mean(axis=0)
    U, S, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    direction = Vt[0]
    direction = direction / np.linalg.norm(direction)
    return centroid.astype(np.float64), direction.astype(np.float64)


def fit_line_ransac(points):
    """
    Fit a 3D line using a PCA-projection + RANSAC per-coordinate approach.
    Returns centroid and unit direction vector, and the boolean inlier mask (len = N).
    """
    # project to 1D parameter t using PCA (principal axis)
    pca = PCA(n_components=1)
    t = pca.fit_transform(points)  # shape (N,1)
    t = t.astype(np.float64)

    # Fit linear models for x,y,z vs t using RANSAC
    lr = LinearRegression()
    ransac_x = RANSACRegressor(lr, residual_threshold=0.02, random_state=0).fit(t, points[:, 0])
    ransac_y = RANSACRegressor(lr, residual_threshold=0.02, random_state=0).fit(t, points[:, 1])
    ransac_z = RANSACRegressor(lr, residual_threshold=0.02, random_state=0).fit(t, points[:, 2])

    # Combined inlier mask (points that all three regressors mark as inliers)
    mask_x = ransac_x.inlier_mask_
    mask_y = ransac_y.inlier_mask_
    mask_z = ransac_z.inlier_mask_
    # If shapes mismatch, fallback to logical_or of available masks
    try:
        inlier_mask = (mask_x & mask_y & mask_z)
    except Exception:
        inlier_mask = (mask_x.astype(bool) & mask_y.astype(bool) & mask_z.astype(bool))

    # Build two sample points on the fitted RANSAC line by sampling min and max t and predicting
    t_min, t_max = t.min(), t.max()
    t_samples = np.array([[t_min], [t_max]], dtype=np.float64)
    x_line = ransac_x.predict(t_samples)
    y_line = ransac_y.predict(t_samples)
    z_line = ransac_z.predict(t_samples)
    p0 = np.array([x_line[0], y_line[0], z_line[0]], dtype=np.float64)
    p1 = np.array([x_line[1], y_line[1], z_line[1]], dtype=np.float64)
    centroid = (p0 + p1) / 2.0
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        # fallback to PCA direction
        centroid_p, dir_p = fit_line_pca(points)
        return centroid_p, dir_p, np.ones(len(points), dtype=bool)
    direction /= norm
    return centroid, direction, inlier_mask


# ---------- Distances & reconstruction ----------

def point_to_line_distance(points, line_point, line_dir):
    """
    Returns orthogonal distances of each point to the line defined by (line_point, line_dir).
    points: (N,3)
    """
    v = points - line_point
    proj = np.dot(v, line_dir)[:, None] * line_dir[None, :]
    orth = v - proj
    d = np.linalg.norm(orth, axis=1)
    return d


def reconstruct_missing_depths_for_mask(mask, depth_arr, intr, line_point, line_dir):
    """
    For each pixel inside mask that has invalid depth, estimate depth by intersecting the camera ray
    with the 3D line (L(s)=line_point + s*line_dir). Solve s*line_dir - t*ray = -line_point for [s,t].
    We use ray = [(x-cx)/fx, (y-cy)/fy, 1.0] (so depth = t).
    Returns a copy of depth_arr with reconstructed depths filled (only modifies masked invalid pixels).
    """
    H, W = depth_arr.shape
    fx, fy, cx, cy = intr
    ys_all, xs_all = np.where(mask == 255)
    zs_all = depth_arr[ys_all, xs_all]

    invalid_idx = np.where(~(np.isfinite(zs_all) & (zs_all > 0.01)))[0]
    if invalid_idx.size == 0:
        return depth_arr.copy(), 0

    depth_recon = depth_arr.copy()
    filled = 0
    for idx in invalid_idx:
        x = int(xs_all[idx]); y = int(ys_all[idx])
        # build ray (not normalized); with our parameterization depth = t
        ray = np.array([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64)
        A = np.column_stack((line_dir, -ray))  # 3x2
        b = -line_point
        # Solve least squares A * [s, t] = b
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            s_val = sol[0]; t_val = sol[1]
            # t_val should be positive (in front of camera). keep only finite positive t
            if np.isfinite(t_val) and (t_val > 0.001):
                depth_recon[y, x] = float(t_val)
                filled += 1
        except Exception:
            pass
    return depth_recon, filled


# ---------- Safe geometry creation ----------

def make_pcd(points, colors=None):
    """
    Safely create an Open3D point cloud with per-point colors (no segfaults).
    """
    if points is None or len(points) == 0:
        print("⚠️ make_pcd(): empty point array — returning empty PointCloud")
        return o3d.geometry.PointCloud()

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        print(f"⚠️ make_pcd(): invalid point shape {pts.shape}, skipping")
        return o3d.geometry.PointCloud()

    mask = np.all(np.isfinite(pts), axis=1)
    if not np.all(mask):
        print(f"make_pcd(): removed {np.sum(~mask)} non-finite points")
        pts = pts[mask]

    if len(pts) == 0:
        print("⚠️ make_pcd(): no valid points after cleaning")
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts))

    if colors is not None:
        cols = np.asarray(colors, dtype=np.float64)
        if cols.shape == pts.shape:
            cols = np.clip(cols, 0.0, 1.0)
        elif cols.shape[0] == mask.shape[0]:
            cols = cols[mask]
            cols = np.clip(cols, 0.0, 1.0)
        else:
            print(f"⚠️ make_pcd(): color shape {cols.shape} doesn't match {pts.shape}")
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
            return pcd

        color_mask = np.all(np.isfinite(cols), axis=1)
        if not np.all(color_mask):
            print(f"make_pcd(): removed {np.sum(~color_mask)} invalid colors")
            cols = cols[color_mask]
            pts = pts[color_mask]
            pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts))
        if len(cols) != len(pts):
            print("⚠️ make_pcd(): color/point length mismatch, painting gray")
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols))
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    return pcd


def make_lineset(line_pts, color=[1.0, 0.0, 0.0]):
    lp = ensure_contiguous_f64(line_pts)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(lp)
    lines_array = ensure_contiguous_i32(np.array([[0, 1]], dtype=np.int32))
    ls.lines = o3d.utility.Vector2iVector(lines_array)
    cols = ensure_contiguous_f64(np.array([color], dtype=np.float64))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


# ---------- Main pipeline ----------

def main():
    print("Loading saved ZED RGB + Depth + Intrinsics...")
    # adjust paths to your files
    rgb_path = os.path.join(CAPTURE_DIR, "rgb_1759998727.png")
    depth_path = os.path.join(CAPTURE_DIR, "depth_1759998727.npy")
    intr_path = os.path.join(CAPTURE_DIR, "intrinsics_1759998727.npy")

    if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(intr_path)):
        print("Files not found. Please place rgb, depth, intrinsics in", CAPTURE_DIR)
        return

    img = cv2.imread(rgb_path)
    if img is None:
        print("Failed to read RGB image:", rgb_path)
        return
    depth_arr = np.load(depth_path)
    intr = np.load(intr_path).astype(np.float64)
    fx, fy, cx, cy = intr

    # show verification
    depth_vis = np.nan_to_num(depth_arr, nan=0.0)
    depth_vis = np.clip(depth_vis, 0.0, np.nanpercentile(depth_vis, 99))
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imshow("RGB (verify)", img)
    cv2.imshow("Depth (verify)", depth_colored)
    print("Press any key to proceed to segmentation...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # segmentation
    seg = BrushSegmenter(img, depth_arr, intr, brush_radius=12)
    mask = seg.run()
    if mask is None:
        print("Segmentation cancelled.")
        return

    # build masked point cloud: returns valid points only
    ys_mask, xs_mask = np.where(mask == 255)
    print(f"Segmented pixels: {len(xs_mask)}")
    if len(xs_mask) == 0:
        print("No segmented pixels.")
        return

    # For mask -> valid/invalid separation:
    zs_mask = depth_arr[ys_mask, xs_mask]
    valid_mask_pixels = np.isfinite(zs_mask) & (zs_mask > 0.01)
    xs_valid = xs_mask[valid_mask_pixels]; ys_valid = ys_mask[valid_mask_pixels]; zs_valid = zs_mask[valid_mask_pixels]

    if len(zs_valid) < 3:
        print("Not enough valid depth points in the segmented area to fit a line.")
        # still attempt to reconstruct? no -> exit
        return

    # compute 3D points for valid segmented pixels
    Xv = (xs_valid - cx) * zs_valid / fx
    Yv = (ys_valid - cy) * zs_valid / fy
    Zv = zs_valid
    pts_valid = np.vstack((Xv, Yv, Zv)).T.astype(np.float64)

    # fit PCA line
    centroid_pca, dir_pca = fit_line_pca(pts_valid)

    # fit RANSAC line
    centroid_ransac, dir_ransac, ransac_inlier_mask = fit_line_ransac(pts_valid)

    # compute inliers by distance (robust fallback)
    dists = point_to_line_distance(pts_valid, centroid_ransac, dir_ransac)
    cloud_extent = np.max(np.linalg.norm(pts_valid - pts_valid.mean(axis=0), axis=1))
    # adaptive threshold: 1% of extent or 0.01 m (whichever is larger)
    dist_thresh = max(0.01, 0.01 * cloud_extent)
    inliers_by_dist = dists <= dist_thresh
    # final inliers: intersection of ransac mask and distance mask -> ensures consistency
    try:
        ransac_mask_final = np.logical_and(ransac_inlier_mask, inliers_by_dist)
    except Exception:
        ransac_mask_final = inliers_by_dist

    print(f"PCA direction: {dir_pca}")
    print(f"RANSAC direction: {dir_ransac}")
    print(f"RANSAC inliers: {np.sum(ransac_mask_final)} / {len(pts_valid)}")

    # Build full point cloud (all valid pixels in the depth map)
    pts_full, colors_full, ys_full, xs_full, idx_map = full_pointcloud_from_depth(depth_arr, intr, img)
    if pts_full is None or len(pts_full) == 0:
        print("No valid points in full depth map.")
        return

    # map segmented valid points to full indices (so we can color them)
    # create mask for line points in full point list:
    line_point_full_mask = np.zeros(len(pts_full), dtype=bool)
    # compute indices of segmented valid pixels in the full cloud (via idx_map)
    full_indices_of_valid_segment = idx_map[ys_valid, xs_valid]  # array of indices into full cloud
    # select those indices which correspond to final ransac inliers
    valid_inlier_full_indices = full_indices_of_valid_segment[ransac_mask_final]
    valid_inlier_full_indices = valid_inlier_full_indices[valid_inlier_full_indices >= 0]
    line_point_full_mask[valid_inlier_full_indices] = True

    # Create Open3D geometries
    pcd_full = make_pcd(pts_full, colors_full)

    # PCA line and RANSAC line Points (endpoints)
    # Determine line points lengths relative to full cloud extent
    extent = np.max(np.linalg.norm(pts_full - pts_full.mean(axis=0), axis=1))
    t_len = max(extent * 2.0, 5.0)
    pca_line_pts = np.vstack((centroid_pca - dir_pca * t_len, centroid_pca + dir_pca * t_len)).astype(np.float64)
    ransac_line_pts = np.vstack((centroid_ransac - dir_ransac * t_len, centroid_ransac + dir_ransac * t_len)).astype(np.float64)
    ls_pca = make_lineset(pca_line_pts, color=[1.0, 0.0, 0.0])       # red
    ls_ransac = make_lineset(ransac_line_pts, color=[0.0, 0.0, 1.0])  # blue

    # Create green point cloud for line points (from full cloud)
    if np.any(line_point_full_mask):
        line_points = pts_full[line_point_full_mask]
        pcd_line = make_pcd(line_points, None)
        pcd_line.paint_uniform_color([0.0, 1.0, 0.0])
    else:
        pcd_line = None

    # Visualize full cloud with both lines and the segmented line points (green)
    geoms = [pcd_full, ls_pca, ls_ransac]
    if pcd_line is not None:
        geoms.append(pcd_line)
    safe_show_open3d(geoms, window_name="Full Cloud with PCA (red) and RANSAC (blue) - segmented line points green")

    # Reconstruct missing depths (only within the segmented mask)
    depth_recon, filled_count = reconstruct_missing_depths_for_mask(mask, depth_arr, intr, centroid_ransac, dir_ransac)
    print(f"Reconstructed {filled_count} previously-missing depths (within the mask).")

    # Build full cloud from reconstructed depths (so more points may now be valid)
    pts_full_recon, colors_full_recon, ys_full_r, xs_full_r, idx_map_recon = full_pointcloud_from_depth(depth_recon, intr, img)
    print(f"Full cloud after reconstruction has {len(pts_full_recon)} valid points (was {len(pts_full)}).")

    # Now mark line points again in reconstructed full cloud using the original segmented pixels mapped onto idx_map_recon
    # For robustness, recompute mapping for all masked pixels and test distance to ransac line
    ys_seg_all, xs_seg_all = np.where(mask == 255)
    # Where depth_recon is finite now:
    valid_after_recon = (np.isfinite(depth_recon[ys_seg_all, xs_seg_all]) & (depth_recon[ys_seg_all, xs_seg_all] > 0.01))
    xs_seg_valid_after = xs_seg_all[valid_after_recon]; ys_seg_valid_after = ys_seg_all[valid_after_recon]
    # get their 3D coords
    zs_after = depth_recon[ys_seg_valid_after, xs_seg_valid_after]
    Xs_after = (xs_seg_valid_after - cx) * zs_after / fx
    Ys_after = (ys_seg_valid_after - cy) * zs_after / fy
    Zs_after = zs_after
    pts_after = np.vstack((Xs_after, Ys_after, Zs_after)).T
    # classify which of these lie close to the RANSAC line
    dists_after = point_to_line_distance(pts_after, centroid_ransac, dir_ransac)
    print("calculated distances after recon")
    thresh_after = max(0.01, 0.01 * np.max(np.linalg.norm(pts_after - pts_after.mean(axis=0), axis=1)))
    print("calculated distances after recon 2")
    on_line_after = dists_after <= thresh_after
    print("calculated distances after recon 3")

    # Make point cloud of fitted points (green) from pts_after[on_line_after]
    if np.any(on_line_after):
        fitted_points_after = pts_after[on_line_after]
        print(fitted_points_after)
        print("made new pcd")
        green_colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(fitted_points_after), 1))
        pcd_fitted = make_pcd(fitted_points_after, green_colors)
    else:
        pcd_fitted = None

    # Visualize reconstructed full cloud (original colors) + fitted points green
    pcd_full_recon = make_pcd(pts_full_recon, colors_full_recon)
    geoms2 = [pcd_full_recon]
    if pcd_fitted is not None:
        geoms2.append(pcd_fitted)
    print("showing pcd")
    safe_show_open3d(geoms2, window_name="Reconstructed Full Cloud (orig colors) + fitted points (green)")

    # Finally: print the 3D coordinate of the lowest segmented point (largest image y)
    ys_seg_pixels, xs_seg_pixels = np.where(mask == 255)
    if len(ys_seg_pixels) == 0:
        print("No segmented pixels to report.")
        return

    # select the pixel with maximum y (lowest on image)
    idx_lowest = np.argmax(ys_seg_pixels)
    x_lowest = int(xs_seg_pixels[idx_lowest]); y_lowest = int(ys_seg_pixels[idx_lowest])
    depth_value = depth_recon[y_lowest, x_lowest]
    if not (np.isfinite(depth_value) and depth_value > 0.0):
        print(f"Lowest segmented pixel at ({x_lowest},{y_lowest}) still has invalid depth after reconstruction.")
    else:
        X_low = (x_lowest - cx) * depth_value / fx
        Y_low = (y_lowest - cy) * depth_value / fy
        Z_low = depth_value
        print(f"Lowest segmented pixel (image coords): ({x_lowest}, {y_lowest})")
        print(f"3D coordinate (meters): [{X_low:.6f}, {Y_low:.6f}, {Z_low:.6f}]")

    # optionally save reconstructed depth
    recon_out = os.path.join(CAPTURE_DIR, "depth_reconstructed.npy")
    try:
        np.save(recon_out, depth_recon)
        print(f"Saved reconstructed depth to: {recon_out}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
