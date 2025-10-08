import cv2
import numpy as np
import open3d as o3d

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
