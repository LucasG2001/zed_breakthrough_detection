import numpy as np
from scipy.optimize import minimize

# --- Helper functions ---

def angles_to_unit_vector(theta_phi):
    theta, phi = theta_phi
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

def ray_point_from_pixel(Kinv, pixel, d):
    uv = np.array([pixel[0], pixel[1], 1.0])
    return d * (Kinv @ uv)

def centroid_of_all(X, s1, s2):
    return (np.sum(X, axis=0) + s1 + s2) / (X.shape[0] + 2)

def perp_residual_sum(u_vec, X, s1, s2, m, weights=None):
    total = 0.0
    if weights is None:
        weights = np.ones(X.shape[0])
    for i, Y in enumerate(X):
        v = Y - m
        perp = v - u_vec * (u_vec.dot(v))
        total += weights[i] * perp.dot(perp)
    for s in (s1, s2):
        v = s - m
        perp = v - u_vec * (u_vec.dot(v))
        total += perp.dot(perp)
    return total

def intersect_ray_with_plane(Kinv, pix, plane_point, plane_normal):
    dir_vec = Kinv @ np.array([pix[0], pix[1], 1.0])
    denom = dir_vec.dot(plane_normal)
    if abs(denom) < 1e-8:
        return None
    return (plane_point.dot(plane_normal)) / denom

# --- Main optimization function ---

def fit_3d_line_with_pixel_constraint_v2(points3d, confidences,
                                         pixel_pt1, pixel_pt2,
                                         fx, fy, cx, cy,
                                         target_distance_mm=145.0,
                                         maxiter=1000, verbose=False):
    """
    Fit a weighted 3D line through 3D points minimizing weighted orthogonal
    distances, subject to the constraint that the two given pixels, when
    backprojected into 3D rays (camera/world frame), have closest points on
    the line exactly `target_distance_mm` apart.

    Parameters
    ----------
    points3d : (N,3) array-like
        Input 3D points in mm.
    confidences : (N,) array-like
        Confidence values, where 0 = perfect, 100 = worst.
    pixel_pt1, pixel_pt2 : (u,v)
        Two pixel coordinates.
    fx, fy, cx, cy : float
        Intrinsic parameters of the camera.
    target_distance_mm : float
        Desired 3D distance between the two projected points on the fitted line.
    maxiter : int
        Maximum iterations for optimizer.

    Returns
    -------
    dict with keys:
      p0 : point on line (3,)
      v  : unit direction (3,)
      proj1, proj2 : two 3D points on the line corresponding to the pixel rays
      projected_distance : distance (mm)
      success, message, fun, res : optimizer info
    """
  
    X = np.asarray(points3d, dtype=float)
    conf = np.asarray(confidences, dtype=float)
    L = target_distance_mm / 1000.0

    # Confidence to weights
    weights = np.clip((100 - conf) / 100.0, 0.0, 1.0)

    # Camera intrinsics
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    Kinv = np.linalg.inv(K)
    pix1 = np.array(pixel_pt1)
    pix2 = np.array(pixel_pt2)

    # Initial direction via PCA
    meanX = X.mean(axis=0)
    C = ((X - meanX).T @ (X - meanX)) / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    u0_vec = eigvecs[:, np.argmax(eigvals)]
    theta0 = np.arccos(np.clip(u0_vec[2], -1, 1))
    phi0 = np.arctan2(u0_vec[1], u0_vec[0])

    # Initial depths: intersect ray with plane through meanX with normal u0_vec
    plane_point = meanX
    plane_normal = u0_vec
    d1_init = intersect_ray_with_plane(Kinv, pix1, plane_point, plane_normal)
    d2_init = intersect_ray_with_plane(Kinv, pix2, plane_point, plane_normal)
    if d1_init is None or d1_init <= 0: d1_init = np.median(X[:,2]) if X.shape[0]>0 else 1.0
    if d2_init is None or d2_init <= 0: d2_init = np.median(X[:,2]) if X.shape[0]>0 else 1.0

    x0 = np.array([theta0, phi0, d1_init, d2_init])

    def constraint_eq(x):
        theta, phi, d1, d2 = x
        u_vec = angles_to_unit_vector([theta, phi])
        s1 = ray_point_from_pixel(Kinv, pix1, d1)
        s2 = ray_point_from_pixel(Kinv, pix2, d2)
        m = centroid_of_all(X, s1, s2)
        t1 = u_vec.dot(s1 - m)
        t2 = u_vec.dot(s2 - m)
        return (t1 - t2)**2 - L**2

    def objective(x):
        theta, phi, d1, d2 = x
        u_vec = angles_to_unit_vector([theta, phi])
        s1 = ray_point_from_pixel(Kinv, pix1, d1)
        s2 = ray_point_from_pixel(Kinv, pix2, d2)
        m = centroid_of_all(X, s1, s2)
        return perp_residual_sum(u_vec, X, s1, s2, m, weights=weights)

    cons = {'type': 'eq', 'fun': constraint_eq}
    bnds = [(1e-6, np.pi-1e-6), (-np.pi, np.pi), (1e-6, None), (1e-6, None)]

    res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                   options={'ftol':1e-9, 'maxiter':maxiter, 'disp':verbose})

    theta_opt, phi_opt, d1_opt, d2_opt = res.x
    u_opt = angles_to_unit_vector([theta_opt, phi_opt])
    s1_opt = ray_point_from_pixel(Kinv, pix1, d1_opt)
    s2_opt = ray_point_from_pixel(Kinv, pix2, d2_opt)
    m_opt = centroid_of_all(X, s1_opt, s2_opt)

    # Convert results back to mm
    return {
        'p0': m_opt,
        'v': u_opt,
        'proj1': s1_opt,
        'proj2': s2_opt,
        'distance between proj points (mm)': np.linalg.norm(s1_opt - s2_opt) * 1000.0,
        'projected_distance': abs(u_opt.dot(s1_opt-m_opt) - u_opt.dot(s2_opt-m_opt)),
        'success': res.success,
        'message': res.message,
        'fun': res.fun,
        'res': res
    }

if __name__ == "__main__":
    # Example usage with dummy data
    points3d = np.array([
        [100.0, 200.0, 50.0],
        [105.0, 210.0, 52.0],
        [110.0, 220.0, 53.5],
        [115.0, 230.0, 54.0],
    ])
    confidences = np.array([0, 10, 30, 60])

    # Intrinsics
    fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0

    # Two pixel points
    pixel_pt1 = (320, 240)  # center pixel
    pixel_pt2 = (420, 260)

    
    result = fit_3d_line_with_pixel_constraint_v2(points3d, confidences,
                                                  pixel_pt1, pixel_pt2,
                                                  fx, fy, cx, cy,
                                                  target_distance_mm=145.0)

    print("✅ Success:", result['success'])
    print("Line point p0:", result['p0'])
    print("Line direction v:", result['v'])
    print("Projected distance (mm):", result['projected_distance'])
