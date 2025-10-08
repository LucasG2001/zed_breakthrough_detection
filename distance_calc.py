"""
zed_distance_measure.py
-----------------------
Click two points in the RGB image to measure their 3D distance (in meters and millimeters)
using the corresponding ZED depth map and intrinsics.
"""

import os
import sys
import cv2
import numpy as np

try:
    import pyzed.sl as sl
except Exception as e:
    print("ZED SDK not found (pyzed.sl missing). Make sure it's installed if using live data.")
    print("Error:", e)

CAPTURE_DIR = "zed_captures"


class TwoPointMeasurer:
    def __init__(self, image, depth_arr, intr):
        self.image = image.copy()
        self.clone = image.copy()
        self.depth_arr = depth_arr
        self.intr = intr
        self.points = []
        self.window = "Click two points to measure distance ('r' to reset, 'q' to quit)"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                cv2.circle(self.clone, (x, y), 5, (0, 0, 255), -1)
                print(f"Point {len(self.points)}: ({x}, {y})")
                if len(self.points) == 2:
                    self.compute_distance()
            else:
                print("Already have two points. Press 'r' to reset.")

    def compute_distance(self):
        (x1, y1), (x2, y2) = self.points
        fx, fy, cx, cy = self.intr

        z1, z2 = self.depth_arr[y1, x1], self.depth_arr[y2, x2]
        if np.isnan(z1) or np.isnan(z2) or z1 <= 0 or z2 <= 0:
            print("Invalid depth values at one or both points.")
            return

        # Backproject to 3D
        X1 = (x1 - cx) * z1 / fx
        Y1 = (y1 - cy) * z1 / fy
        X2 = (x2 - cx) * z2 / fx
        Y2 = (y2 - cy) * z2 / fy

        p1 = np.array([X1, Y1, z1]) * 0.001
        p2 = np.array([X2, Y2, z2]) * 0.001

        dist_m = np.linalg.norm(p1 - p2)
        dist_mm = dist_m * 1000.0

        print(f"\n--- Distance Results ---")
        print(f"Point 1 (pixels): ({x1}, {y1})  → 3D: {p1}")
        print(f"Point 2 (pixels): ({x2}, {y2})  → 3D: {p2}")
        print(f"Distance: {dist_m:.4f} m   |   {dist_mm:.2f} mm\n")

        cv2.line(self.clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(self.window, self.clone)

    def run(self):
        while True:
            cv2.imshow(self.window, self.clone)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.clone = self.image.copy()
                self.points = []
                print("Reset points.")
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()


def main():
    print("Starting 3D distance measurement (loading saved data)")

    # --- Load saved images ---
    rgb_path = os.path.join(CAPTURE_DIR, "rgb_1759840630.png")
    depth_path = os.path.join(CAPTURE_DIR, "depth_1759840630.npy")
    intr_path = os.path.join(CAPTURE_DIR, "intrinsics_1759840630.npy")

    # Load data
    img = cv2.imread(rgb_path)
    if img is None:
        print(f"Error: could not read {rgb_path}")
        return

    try:
        depth_arr = np.load(depth_path)
        intr = np.load(intr_path)
    except Exception as e:
        print("Failed to load depth or intrinsics:", e)
        return

    # --- Show for confirmation ---
    depth_vis = depth_arr.copy()
    depth_vis[np.isnan(depth_vis)] = 0
    depth_vis = np.clip(depth_vis, 0, np.nanpercentile(depth_vis, 99))
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("RGB Image", img)
    cv2.imshow("Depth Image (colored)", depth_colored)
    print("Press any key to start distance measurement...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    measurer = TwoPointMeasurer(img, depth_arr, intr)
    measurer.run()


if __name__ == "__main__":
    main()
