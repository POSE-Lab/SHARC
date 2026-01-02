import numpy as np
from scipy.spatial import KDTree
from sharc.math.sh import generate_candidate_points_vec

def filter_points_by_knn(ref_points, coeffs, N_recon, lmax, k=5):

    # 1. Generate candidate points (intial point-cloud)
    candidate_points, generating_indices = generate_candidate_points_vec(
        ref_points, coeffs, N_recon, lmax
    )
    
    if candidate_points.shape[0] == 0:
        print("Warning: No candidate points were generated.")
        return np.array([]).reshape(0, 3), np.array([])
        
    N_pts = candidate_points.shape[0]

    # 2. Build KD-Tree and find the K-nearest reference points
    ref_point_tree = KDTree(ref_points)
    print(f"Querying k-neighbors (k={k}) for {N_pts} candidates...")
    try:
        _, neighbor_indices = ref_point_tree.query(candidate_points, k=k)
    except ValueError as e:
        k = min(k, len(ref_points))
        if k == 0: return np.array([]).reshape(0, 3), np.array([])
        _, neighbor_indices = ref_point_tree.query(candidate_points, k=k)

    if k == 1 and len(neighbor_indices.shape) == 1:
        neighbor_indices = neighbor_indices.reshape(-1, 1)

    # 3. Keep points from local generators
    generator_in_knn = np.any(neighbor_indices == generating_indices[:, np.newaxis], axis=1)
    keep_mask = generator_in_knn
    
    # 4. Final filtered point-cloud
    filtered_points = candidate_points[keep_mask]
    filtered_indices = generating_indices[keep_mask]
    print(f"Finished filtering. Kept {len(filtered_points)} / {N_pts} points.")
    
    return filtered_points, filtered_indices