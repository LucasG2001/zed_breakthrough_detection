"""
zed_line_fit_reconstruct.py
--------------------------------
Segment 3D points from an RGB+Depth pair, fit a line (PCA + RANSAC),
visualize the fit, and reconstruct missing depth values by projecting
invalid points onto the fitted line.
"""

import os
import cv2
import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA

CAPTURE_DIR = "zed_captures"


# --------------------------- Utility Functions ---------------------------

def mask_to_pointcloud(mask, depth_arr, intr, rgb_img=None):
    """Convert masked pixels to 3D point cloud."""
    fx, fy, cx, cy = intr
    ys, xs = np.where(mask == 255)
    zs = depth_arr[ys, xs]
    valid = np.isfinite(zs) & (zs > 0.01)
    xs_valid, ys_valid, zs_valid = xs[valid], ys[valid], zs[valid]

    if len(zs_valid) == 0:
        return None, None, None, None

    X = (xs_valid - cx) * zs_valid / fx
    Y = (ys_valid - cy) * zs_valid / fy
    Z = zs_valid
    pts = np.vstack((X, Y, Z)).T

    colors = None
    if rgb_img is not None:
        rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        colors = rgb[ys_valid, xs_valid] / 255.0

    # Return also the invalid ones for later reconstruction
    xs_invalid, ys_invalid = xs[~valid], ys[~valid]
    return pts, colors, (xs_invalid, ys_invalid), (xs_valid, ys_valid)


def fit_line_pca(points):
    """Fit line using PCA (SVD-based)."""
    centroid = points.mean(axis=0)
    pts_centered = points - centroid
    U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    direction = Vt[0]
    return centroid, direction


def fit_line_ransac(points):
    """Fit a line using RANSAC (3D simplified to one coordinate at a time)."""
    X = points[:, 0].reshape(-1, 1)
    Y = points[:, 1]
    Z = points[:, 2]

    ransac_y = RANSACRegressor().fit(X, Y)
    ransac_z = RANSACRegressor().fit(X, Z)

    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = ransac_y.predict(x_line.reshape(-1, 1))
    z_line = ransac_z.predict(x_line.reshape(-1, 1))

    centroid = np.array([X.mean(), Y.mean(), Z.mean()])
    direction = np.array([1.0, np.mean(np.gradient(y_line)), np.mean(np.gradient(z_line))])
    direction /= np.linalg.norm(direction)

    return centroid, direction


def visualize_with_open3d(points, centroid, direction, colors=None):
    """Visualize 3D points with their original RGB colors and a red fitted line."""
    # --- Clean up NaNs/Infs ---
    mask = np.all(np.isfinite(points), axis=1)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]
        color_mask = np.all(np.isfinite(colors), axis=1)
        if not np.all(color_mask):
            points = points[color_mask]
            colors = colors[color_mask]

    if len(points) == 0:
        print("No valid 3D points to visualize.")
        return

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # --- Create colored point cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None and len(colors) == len(points):
        colors = np.clip(colors.astype(np.float64), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # --- Compute line geometry ---
    if not (np.all(np.isfinite(centroid)) and np.all(np.isfinite(direction))):
        centroid = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
        direction = Vt[0]

    extent = np.max(np.linalg.norm(points - centroid, axis=1))
    t = max(extent * 2.0, 5.0)
    line_pts = np.array([centroid - direction * t, centroid + direction * t], dtype=np.float64)

    if not np.all(np.isfinite(line_pts)):
        print("Error: invalid line points, skipping line visualization.")
        o3d.visualization.draw_geometries([pcd], window_name="3D Visualization")
        return

    # --- Safe LineSet construction ---
    line_set = o3d.geometry.LineSet()
    line_pts_clean = np.ascontiguousarray(line_pts, dtype=np.float64)
    line_set.points = o3d.utility.Vector3dVector(line_pts_clean)

    lines_array = np.array([[0, 1]], dtype=np.int32)
    line_set.lines = o3d.utility.Vector2iVector(lines_array)

    colors_array = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    line_set.colors = o3d.utility.Vector3dVector(colors_array)

    print("Created line set — ready to visualize")

    # --- Visualize both ---
    o3d.visualization.draw_geometries(
        [pcd, line_set],
        window_name="3D Fit Visualization",
        width=960,
        height=720,
    )


def reconstruct_missing_depth(xs_invalid, ys_invalid, centroid, direction, intr):
    """Estimate missing depth values by projecting onto the fitted 3D line."""
    fx, fy, cx, cy = intr
    reconstructed_depths = []

    for x, y in zip(xs_invalid, ys_invalid):
        # Compute backprojected ray direction
        ray_dir = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
        ray_dir /= np.linalg.norm(ray_dir)

        # Solve for intersection between ray and line
        A = np.stack([direction, -ray_dir], axis=1)
        try:
            t_vals, _, _, _ = np.linalg.lstsq(A, centroid, rcond=None)
            depth = t_vals[0]
        except np.linalg.LinAlgError:
            depth = np.nan
        reconstructed_depths.append(depth)

    return np.array(reconstructed_depths)


# --------------------------- Segmentation UI ---------------------------

class BrushSegmenter:
    def __init__(self, image, depth_arr, intr):
        self.image = image.copy()
        self.clone = image.copy()
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.brush_size = max(3, int(0.6 * 10))  # Smaller brush
        self.drawing = False
        self.depth_arr = depth_arr
        self.intr = intr
        self.window = "Draw mask - hold left mouse, 'f' finalize, 'r' reset, 'q' quit"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            cv2.circle(self.clone, (x, y), self.brush_size, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def reset(self):
        self.clone = self.image.copy()
        self.mask[:] = 0

    def run(self):
        while True:
            cv2.imshow(self.window, self.clone)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('r'):
                self.reset()
            elif key == ord('f'):
                print("Mask finalized — computing 3D points...")
                pts, colors, invalid, valid = mask_to_pointcloud(self.mask, self.depth_arr, self.intr, self.image)
                if pts is None:
                    print("No valid 3D points.")
                    continue

                print(f"Valid 3D points: {len(pts)}, Invalid pixels: {len(invalid[0])}")

                centroid_pca, direction_pca = fit_line_pca(pts)
                centroid_ransac, direction_ransac = fit_line_ransac(pts)

                print("PCA line direction:", direction_pca)
                print("RANSAC line direction:", direction_ransac)

                visualize_with_open3d(pts, centroid_pca, direction_pca, colors)

                print("Reconstructing missing depth values...")
                reconstructed = reconstruct_missing_depth(invalid[0], invalid[1], centroid_pca, direction_pca, self.intr)
                print(f"Reconstructed {np.sum(np.isfinite(reconstructed))}/{len(reconstructed)} missing depths.")
                break

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()


# --------------------------- Main Entry ---------------------------

def main():
    print("Starting 3D segmentation + line fitting + depth reconstruction")

    rgb_path = os.path.join(CAPTURE_DIR, "rgb_1759840630.png")
    depth_path = os.path.join(CAPTURE_DIR, "depth_1759840630.npy")
    intr_path = os.path.join(CAPTURE_DIR, "intrinsics_1759840630.npy")

    img = cv2.imread(rgb_path)
    if img is None:
        print(f"Error loading {rgb_path}")
        return

    depth_arr = np.load(depth_path)
    intr = np.load(intr_path)

    # Visual check
    depth_vis = depth_arr.copy()
    depth_vis[np.isnan(depth_vis)] = 0
    depth_vis = np.clip(depth_vis, 0, np.nanpercentile(depth_vis, 99))
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("RGB", img)
    cv2.imshow("Depth", depth_colored)
    print("Press any key to start segmentation...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    seg = BrushSegmenter(img, depth_arr, intr)
    seg.run()


if __name__ == "__main__":
    main()
