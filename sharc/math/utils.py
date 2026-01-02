import open3d as o3d
import numpy as np

def _safe_normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n
    
def estimate_outward_normals(points, origins_per_point,
                             max_nn=30,
                             radius=None,
                             use_hybrid=True):
   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Auto-pick a reasonable radius from data if not provided:
    # Use 1% of bbox diagonal as a baseline, then inflate a bit.
    if radius is None:
        aabb = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(aabb.get_extent())
        # If points are dense, 1% of diag is often good; multiply a bit for stability.
        radius = max(1e-6, 0.015 * diag)

    if use_hybrid:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn)
        )

    normals = np.asarray(pcd.normals)
    points_np = np.asarray(pcd.points)

    # Ray directions: from origin (skeleton) to point
    ray_dirs = _safe_normalize(points_np - origins_per_point)

    # Flip normals to agree with ray directions (i.e., outward from skeleton)
    flip_mask = (np.sum(normals * ray_dirs, axis=1) < 0.0)
    normals[flip_mask] *= -1.0

    # Optional: light smoothing via normalize again (already unit, but keep clean)
    normals = _safe_normalize(normals)
    return normals