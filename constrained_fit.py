import numpy as np
from scipy.optimize import minimize

# --- Helpers (same conventions as your file) ---

def angles_to_unit_vector(theta_phi):
    theta, phi = theta_phi
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

def ray_point_from_pixel(Kinv, pixel, d):
    uv = np.array([pixel[0], pixel[1], 1.0])
    return d * (Kinv @ uv)

def perp_residual_sum(u_vec, X, m, weights=None):
    if weights is None:
        weights = np.ones(X.shape[0])
    v = X - m  # (N,3)
    # perpendicular component of each v to u_vec:
    proj_along = (v @ u_vec)[:, None] * u_vec[None, :]
    perp = v - proj_along
    # squared norms weighted
    return float(np.sum(weights * np.sum(perp**2, axis=1)))

# --- Main solver with hard intersection + distance constraints ---

def fit_3d_line_with_pixel_constraint_v2(points3d, confidences,
                                              pixel_pt1, pixel_pt2,
                                              fx, fy, cx, cy,
                                              target_distance_mm=145.0,
                                              maxiter=1000, verbose=False):
    """
    Fit a 3D line that:
      - intersects the two pixel rays (hard equality constraints),
      - the two intersection points on the line are exactly target_distance_mm apart,
      - minimizes orthogonal distances from points3d to the line (weighted).
    Returns dict with line params and intersection points.
    """
    X = np.asarray(points3d, dtype=float)
    conf = np.asarray(confidences, dtype=float)
    L = target_distance_mm/1000  # keep units consistent with X (assume X in m)
    # weights: transform confidences (same as you used)
    weights = 100.0 - confidences
    weights = np.clip(weights, 0, None)  # no negative weights
    weights = weights / np.sum(weights)  # normalize to sum=1

    # camera intrinsics
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    Kinv = np.linalg.inv(K)
    pix1 = np.array(pixel_pt1)
    pix2 = np.array(pixel_pt2)

    # initial guess for line via PCA
    meanX = X.mean(axis=0)
    C = ((X - meanX).T @ (X - meanX)) / (X.shape[0] if X.shape[0] > 0 else 1.0)
    eigvals, eigvecs = np.linalg.eigh(C)
    u0_vec = eigvecs[:, np.argmax(eigvals)]
    theta0 = np.arccos(np.clip(u0_vec[2], -1.0, 1.0))
    phi0 = np.arctan2(u0_vec[1], u0_vec[0])

    # initial depths: intersect ray with plane (plane: meanX, normal u0_vec)
    def intersect_ray_with_plane(Kinv, pix, plane_point, plane_normal):
        dir_vec = Kinv @ np.array([pix[0], pix[1], 1.0])
        denom = dir_vec.dot(plane_normal)
        if abs(denom) < 1e-8:
            return None
        # plane equation: plane_normal . (X) = plane_normal . plane_point
        return (plane_point.dot(plane_normal)) / denom

    d1_init = intersect_ray_with_plane(Kinv, pix1, meanX, u0_vec)
    d2_init = intersect_ray_with_plane(Kinv, pix2, meanX, u0_vec)
    if d1_init is None or d1_init <= 0: d1_init = max(1.0, meanX[2])
    if d2_init is None or d2_init <= 0: d2_init = max(1.0, meanX[2])

    s1_init = ray_point_from_pixel(Kinv, pix1, d1_init)
    s2_init = ray_point_from_pixel(Kinv, pix2, d2_init)
    # project these onto initial line (m = meanX)
    t1_init = float(u0_vec.dot(s1_init - meanX))
    t2_init = float(u0_vec.dot(s2_init - meanX))
    # if t2 - t1 not matching L, nudge them symmetric about mean
    mid = 0.5 * (t1_init + t2_init)
    halfL = 0.5 * (L if L != 0 else 1.0)
    t1_init = mid - halfL
    t2_init = mid + halfL

    # x = [theta, phi, m_x, m_y, m_z, d1, d2, t1, t2]
    x0 = np.hstack([theta0, phi0, meanX, d1_init, d2_init, t1_init, t2_init])

    # Constraints: vector-valued equality with length 7
    def constraint_vector(x):
        theta, phi = x[0], x[1]
        m = np.array(x[2:5])
        d1, d2 = x[5], x[6]
        t1, t2 = x[7], x[8]
        u_vec = angles_to_unit_vector([theta, phi])

        s1 = ray_point_from_pixel(Kinv, pix1, d1)   # point on ray1
        s2 = ray_point_from_pixel(Kinv, pix2, d2)   # point on ray2
        p1 = m + t1 * u_vec                         # point on line
        p2 = m + t2 * u_vec

        # ray-line intersection enforced by s1 == p1 and s2 == p2 (3+3 eqns)
        c1 = s1 - p1   # shape (3,)
        c2 = s2 - p2   # shape (3,)

        # distance along line constraint: (t2 - t1)^2 - L^2 == 0
        c3 = np.array([(t2 - t1)**2 - L**2])

        return np.hstack([c1, c2, c3])  # length 7

    # objective: perpendicular residuals to the line (m, u)
    def objective(x):
        theta, phi = x[0], x[1]
        m = np.array(x[2:5])
        u_vec = angles_to_unit_vector([theta, phi])
        return perp_residual_sum(u_vec, X, m, weights=weights)

    # bounds: keep sensible ranges (depths > 0)
    bnds = [
        (1e-6, np.pi - 1e-6),   # theta
        (-np.pi, np.pi),        # phi
        (None, None), (None, None), (None, None),  # m_x, m_y, m_z
        (1e-6, None), (1e-6, None),  # d1, d2 (positive depths)
        (None, None), (None, None)   # t1, t2
    ]

    cons = {'type': 'eq', 'fun': constraint_vector}

    #res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
    #               options={'ftol': 1e-9, 'maxiter': maxiter, 'disp': verbose})

    res = minimize(
    objective, x0, method='trust-constr',
    constraints=cons, bounds=bnds,
    options={'verbose': 2, 'maxiter': maxiter}
)
    
    theta_opt, phi_opt = res.x[0], res.x[1]
    m_opt = res.x[2:5]
    d1_opt, d2_opt = res.x[5], res.x[6]
    t1_opt, t2_opt = res.x[7], res.x[8]
    u_opt = angles_to_unit_vector([theta_opt, phi_opt])

    s1_opt = ray_point_from_pixel(Kinv, pix1, d1_opt)
    s2_opt = ray_point_from_pixel(Kinv, pix2, d2_opt)
    p1_opt = m_opt + t1_opt * u_opt
    p2_opt = m_opt + t2_opt * u_opt

    return {
        'p0': m_opt,
        'v': u_opt,
        'ray1_depth': d1_opt,
        'ray2_depth': d2_opt,
        'ray1_point_on_ray': s1_opt,
        'ray2_point_on_ray': s2_opt,
        'proj1_on_line': p1_opt,
        'proj2_on_line': p2_opt,
        'distance_between_proj_points_mm': np.linalg.norm(p2_opt - p1_opt),
        'distance_between_ray_points_mm': np.linalg.norm(s2_opt - s1_opt),
        'success': res.success,
        'message': res.message,
        'fun': res.fun,
        'res': res
    }

# --- Example usage (replace with your big dataset) ---
if __name__ == "__main__":
    # toy data (replace with your data in mm)
    points3d = np.array([
        [100.0, 200.0, 50.0],
        [105.0, 210.0, 52.0],
        [110.0, 220.0, 53.5],
        [115.0, 230.0, 54.0],
    ])
    confidences = np.array([0, 10, 30, 60])

    fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
    pixel_pt1 = (320, 240)
    pixel_pt2 = (420, 260)

    out = fit_3d_line_with_pixel_constraint_v2(points3d, confidences,
                                                    pixel_pt1, pixel_pt2,
                                                    fx, fy, cx, cy,
                                                    target_distance_mm=145.0,
                                                    maxiter=1000, verbose=True)
    print("Success:", out['success'], out['message'])
    print("Line point p0:", out['p0'])
    print("Line dir v:", out['v'])
    print("Projection pts on line:", out['proj1_on_line'], out['proj2_on_line'])
    print("Distance between proj points (mm):", out['distance_between_proj_points_mm'])
    print("Ray points (should equal proj points up to small tol):", out['ray1_point_on_ray'], out['ray2_point_on_ray'])
