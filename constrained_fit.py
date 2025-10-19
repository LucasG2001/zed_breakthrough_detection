import numpy as np
from scipy.optimize import minimize

# --- Helpers ---

def ray_point_from_pixel(Kinv, pixel, d):
    """Get 3D point at depth d along the ray from pixel."""
    uv = np.array([pixel[0], pixel[1], 1.0])
    return d * (Kinv @ uv)

def ray_direction_from_pixel(Kinv, pixel):
    """Get normalized ray direction from pixel."""
    uv = np.array([pixel[0], pixel[1], 1.0])
    ray_dir = Kinv @ uv
    return ray_dir / np.linalg.norm(ray_dir)

def closest_point_ray_to_line(ray_origin, ray_dir, line_point, line_dir):
    """
    Find closest points between a ray and a line.
    Returns: (point_on_ray, point_on_line, distance, ray_depth)
    """
    # Parametric forms:
    # Ray: P_r(s) = ray_origin + s * ray_dir, s >= 0
    # Line: P_l(t) = line_point + t * line_dir
    
    w = ray_origin - line_point
    a = np.dot(ray_dir, ray_dir)
    b = np.dot(ray_dir, line_dir)
    c = np.dot(line_dir, line_dir)
    d = np.dot(ray_dir, w)
    e = np.dot(line_dir, w)
    
    denom = a * c - b * b
    
    if abs(denom) < 1e-10:
        # Parallel case
        s = 0.0
        t = e / c if abs(c) > 1e-10 else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    
    # Clamp s to be non-negative (ray constraint)
    s = max(0.0, s)
    
    point_on_ray = ray_origin + s * ray_dir
    point_on_line = line_point + t * line_dir
    dist = np.linalg.norm(point_on_ray - point_on_line)
    
    return point_on_ray, point_on_line, dist, s

def perp_residual_sum(u_vec, X, m, weights=None):
    """Sum of squared perpendicular distances from points X to line (m, u_vec)."""
    if weights is None:
        weights = np.ones(X.shape[0])
    v = X - m  # (N,3)
    proj_along = (v @ u_vec)[:, None] * u_vec[None, :]
    perp = v - proj_along
    return float(np.sum(weights * np.sum(perp**2, axis=1)))

# --- Main solver ---

def fit_3d_line_with_pixel_constraint_v3(points3d, m_init, u_init, confidences,
                                         pixel_pt1, pixel_pt2,
                                         fx, fy, cx, cy,
                                         target_distance_mm=122.0,
                                         maxiter=2000, verbose=True):
    """
    Fit a 3D line with 6 optimization variables: [u_x, u_y, u_z, m_x, m_y, m_z]
    
    Uses direct direction vector representation (more linear than angles).
    Direction is normalized internally to maintain unit length.
    
    Objective:
      - 70% weight: Minimize perpendicular distance from points3d to line (weighted)
      - 30% weight: Minimize distance from rays to the line
    
    Constraints (1 hard constraint):
      - Two points on the line (closest to the rays) are exactly L apart
      
    Returns dict with line params and the two constraint points on the line.
    """
    X = np.asarray(points3d, dtype=float)
    conf = np.asarray(confidences, dtype=float)
    L = target_distance_mm / 1000.0  # convert to meters
    
    # Weights for data points: transform confidences
    data_weights = 100.0 - conf
    data_weights = np.clip(data_weights, 0, None)
    total_data_weight = np.sum(data_weights)
    data_weights = data_weights / (total_data_weight + 1e-10)  # normalize to sum=1

    # Camera intrinsics
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    Kinv = np.linalg.inv(K)
    pix1 = np.array(pixel_pt1, dtype=float)
    pix2 = np.array(pixel_pt2, dtype=float)
    
    # Ray directions (normalized)
    ray_dir1 = ray_direction_from_pixel(Kinv, pix1)
    ray_dir2 = ray_direction_from_pixel(Kinv, pix2)
    ray_origin = np.array([0.0, 0.0, 0.0])  # camera at origin

    # Initial guess: [u_x, u_y, u_z, m_x, m_y, m_z]
    u_init_normalized = u_init / np.linalg.norm(u_init)
    x0 = np.hstack([u_init_normalized, m_init])

    def get_line_params(x):
        """Extract and normalize line parameters from optimization vector."""
        u_vec = np.array(x[0:3])
        u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-10)  # normalize to unit vector
        m = np.array(x[3:6])
        return m, u_vec

    def constraint_vector(x):
        """
        One constraint: Distance between the two closest points on the line = L
        """
        m, u_vec = get_line_params(x)
        
        # Find closest points on line to each ray
        _, p1_on_line, _, _ = closest_point_ray_to_line(ray_origin, ray_dir1, m, u_vec)
        _, p2_on_line, _, _ = closest_point_ray_to_line(ray_origin, ray_dir2, m, u_vec)
        
        # Constraint: distance between these points = L
        dist = np.linalg.norm(p2_on_line - p1_on_line)
        c_dist = dist - L
        
        return np.array([c_dist])

    def objective(x):
        """
        Minimize weighted combination:
        - 70% weight: perpendicular distance from data points to line
        - 30% weight: distance from rays to line
        
        Weight balance ensures ray constraint is significant but not dominant.
        """
        m, u_vec = get_line_params(x)
        
        # Primary term: perpendicular distance from data points (weighted by confidence)
        data_residual = perp_residual_sum(u_vec, X, m, weights=data_weights)
        
        # Secondary term: distance from rays to line
        _, _, dist1, _ = closest_point_ray_to_line(ray_origin, ray_dir1, m, u_vec)
        _, _, dist2, _ = closest_point_ray_to_line(ray_origin, ray_dir2, m, u_vec)
        ray_residual = dist1**2 + dist2**2
        
        # Weight calculation:
        # We want ray term to be ~30% of total, data term ~70%
        # If data_residual has scale ~1 (normalized weights), 
        # then we want: 0.7 * data_residual + 0.3 * (scaled ray_residual)
        # But ray_residual is in m^2, so we scale it to match data term magnitude
        
        # Estimate typical scale: assume data points form ~0.1m cloud
        # Then data_residual ~ 0.01 m^2, ray_residual ~ 0.001 m^2
        # Weight ratio: (0.3/0.7) * (data_residual_scale / ray_residual_scale)
        
        # Simpler approach: weight ray term such that at initialization,
        # it contributes ~30% to total objective
        ray_weight = 0.3 / 0.7  # ratio of 30:70
        
        return data_residual + ray_weight * ray_residual

    # Bounds: keep direction components reasonable (will be normalized anyway)
    bnds = [
        (-10, 10), (-10, 10), (-10, 10),  # u_x, u_y, u_z (will be normalized)
        (None, None), (None, None), (None, None),  # m_x, m_y, m_z
    ]

    cons = {'type': 'eq', 'fun': constraint_vector}

    # Check initial constraint violation
    init_constraint = constraint_vector(x0)
    if verbose:
        print(f"Initial constraint violation: {np.linalg.norm(init_constraint):.6f}")
        m_init_check, u_init_check = get_line_params(x0)
        init_obj = objective(x0)
        print(f"Initial objective: {init_obj:.6e}")

    # Optimize - SLSQP is good for this type of problem (smooth, small number of constraints)
    res = minimize(
        objective, x0, method='SLSQP',
        constraints=cons, bounds=bnds,
        options={'disp': verbose, 'maxiter': maxiter, 'ftol': 1e-9}
    )
    
    # If SLSQP fails, try trust-constr as backup
    if not res.success and verbose:
        print("\nSLSQP failed, trying trust-constr...")
        res = minimize(
            objective, x0, method='trust-constr',
            constraints=cons, bounds=bnds,
            options={'verbose': 1 if verbose else 0, 'maxiter': maxiter}
        )
    
    # Extract results
    m_opt, u_opt = get_line_params(res.x)
    
    # Get the two points on the line (closest to rays)
    ray1_pt, p1_on_line, dist1, d1_opt = closest_point_ray_to_line(
        ray_origin, ray_dir1, m_opt, u_opt
    )
    ray2_pt, p2_on_line, dist2, d2_opt = closest_point_ray_to_line(
        ray_origin, ray_dir2, m_opt, u_opt
    )
    
    # Compute t-parameters for the line points
    t1 = np.dot(p1_on_line - m_opt, u_opt)
    t2 = np.dot(p2_on_line - m_opt, u_opt)
    
    return {
        'p0': m_opt,
        'v': u_opt,
        'ray1_depth': d1_opt,
        'ray2_depth': d2_opt,
        'ray1_point_on_ray': ray1_pt,
        'ray2_point_on_ray': ray2_pt,
        'proj1_on_line': p1_on_line,
        'proj2_on_line': p2_on_line,
        't1': t1,
        't2': t2,
        'distance_between_line_points_m': np.linalg.norm(p2_on_line - p1_on_line),
        'distance_between_line_points_mm': np.linalg.norm(p2_on_line - p1_on_line) * 1000,
        'ray1_to_line_distance_mm': dist1 * 1000,
        'ray2_to_line_distance_mm': dist2 * 1000,
        'success': res.success,
        'message': res.message,
        'objective_value': res.fun,
        'constraint_violation': np.linalg.norm(constraint_vector(res.x)),
        'res': res
    }


# --- Example usage ---
if __name__ == "__main__":
    # Toy data in meters
    points3d = np.array([
        [0.100, 0.200, 0.500],
        [0.105, 0.210, 0.520],
        [0.110, 0.220, 0.535],
        [0.115, 0.230, 0.540],
    ])
    confidences = np.array([0, 10, 30, 60])

    # Initial guess
    m_init = points3d.mean(axis=0)
    u_init = np.array([1.0, 1.0, 0.1])
    u_init = u_init / np.linalg.norm(u_init)

    fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
    pixel_pt1 = (320, 240)
    pixel_pt2 = (420, 260)

    out = fit_3d_line_with_pixel_constraint_v3(
        points3d, m_init, u_init, confidences,
        pixel_pt1, pixel_pt2,
        fx, fy, cx, cy,
        target_distance_mm=122.0,
        maxiter=2000, verbose=True
    )
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Success: {out['success']}")
    print(f"Message: {out['message']}")
    print(f"\nLine point p0: {out['p0']}")
    print(f"Line direction v: {out['v']}")
    print(f"Direction magnitude: {np.linalg.norm(out['v'])}")
    print(f"\nPoint 1 on line: {out['proj1_on_line']}")
    print(f"Point 2 on line: {out['proj2_on_line']}")
    print(f"\nDistance between line points: {out['distance_between_line_points_mm']:.3f} mm (target: 122.0 mm)")
    print(f"Ray 1 to line distance: {out['ray1_to_line_distance_mm']:.6f} mm")
    print(f"Ray 2 to line distance: {out['ray2_to_line_distance_mm']:.6f} mm")
    print(f"\nDepths: d1={out['ray1_depth']:.4f} m, d2={out['ray2_depth']:.4f} m")
    print(f"Objective value: {out['objective_value']:.6e}")
    print(f"Constraint violation: {out['constraint_violation']:.6e}")