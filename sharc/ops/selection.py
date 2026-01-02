from typing import Union, Tuple
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial.distance import cdist
from sharc.utils.warp_utils import WarpMeshSampler

def compute_skeleton_points_visibility(
    mesh: Union[trimesh.Trimesh, o3d.geometry.TriangleMesh],
    init_skeleton_points: np.ndarray = None,
    surface_samples: int = 10000,
    proximity_threshold: float = 0.1,
    uniformity_weight: float = 0.7,
    centrality_weight: float = 0.5,
    max_points: int = None,
    random_sample_number: int = 100000,
    min_distance_from_surface: float = 1e-5,
    generate_candidates: bool = False,
    save_candidates_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute skeleton points using visibility-based selection method.
    
    This function selects a compact set of skeleton points from initial candidates
    based on their visibility and coverage of the surface. The selection process
    iteratively chooses points that can see uncovered surface regions while
    maintaining spatial uniformity and favoring interior points.
    
    Args:
        mesh: Input 3D mesh (trimesh or Open3D format)
        init_skeleton_points: (n, 3) array of initial candidate skeleton points.
                             If None and generate_candidates=True, will randomly 
                             sample points inside the mesh
        surface_samples: Number of points to sample on the mesh surface
        proximity_threshold: Max distance for a point to be considered "covered".
                           For models scaled to unit sphere (diameter ~2.0):
                           - 0.05-0.1: tight coverage, many skeleton points
                           - 0.1-0.2: moderate coverage (recommended)
                           - 0.2-0.5: loose coverage, fewer skeleton points
        uniformity_weight: Weight for spatial uniformity (0-1). Higher values
                          favor points that are far from already-selected points
        centrality_weight: Weight for centrality/interiority (0-1+). Higher values
                          favor points that are farther from the surface (more interior).
                          Recommended range: 0.3-1.0
        max_points: Maximum number of skeleton points to select. If None, 
                   continues until all coverable points are covered
        random_sample_number: Number of random points to sample inside mesh 
                             if generate_candidates=True
        generate_candidates: If True and init_skeleton_points is None, will
                           randomly sample candidate points inside the mesh
        save_candidates_path: Optional path to save the initial candidate points
                             as a PLY file (e.g., "output/candidates.ply")
    
    Returns:
        Tuple containing:
            - selected_skeleton_points: (k, 3) array of selected skeleton points
            - selected_indices: (k,) array of indices into init_skeleton_points
    
    Example:
        >>> mesh = trimesh.load("model.obj")
        >>> # Option 1: Use provided candidates
        >>> candidates = ... # (n, 3) array
        >>> skeleton_points, indices = compute_skeleton_points_visibility(
        ...     mesh, candidates, proximity_threshold=0.15, 
        ...     centrality_weight=0.5, max_points=30
        ... )
        >>> 
        >>> # Option 2: Generate candidates randomly
        >>> skeleton_points, indices = compute_skeleton_points_visibility(
        ...     mesh, generate_candidates=True, random_sample_number=100000,
        ...     proximity_threshold=0.15, uniformity_weight=0.7,
        ...     centrality_weight=0.5, max_points=30
        ... )
    """
    # Convert trimesh to Open3D if needed
    if isinstance(mesh, trimesh.Trimesh):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
        trimesh_mesh = mesh
    else:
        o3d_mesh = mesh
        # Convert Open3D to trimesh for candidate generation
        trimesh_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles)
        )
    
    # Generate candidate points if requested
    if init_skeleton_points is None and generate_candidates:
        print(f"Generating {random_sample_number} random candidate points inside mesh...")
        init_skeleton_points = _generate_random_interior_points(
            trimesh_mesh, random_sample_number, min_distance_to_surface=min_distance_from_surface
        )
        print(f"Generated {len(init_skeleton_points)} interior candidate points.")
    elif init_skeleton_points is None:
        raise ValueError(
            "Either provide init_skeleton_points or set generate_candidates=True"
        )
    
    # Save candidate points if path is provided
    if save_candidates_path is not None:
        print(f"Saving {len(init_skeleton_points)} candidate points to: {save_candidates_path}")
        candidate_pcd = o3d.geometry.PointCloud()
        candidate_pcd.points = o3d.utility.Vector3dVector(init_skeleton_points)
        o3d.io.write_point_cloud(save_candidates_path, candidate_pcd)
        print(f"Candidate points saved successfully.")
    
    # Sample points on mesh surface uniformly
    print(f"Sampling {surface_samples} points on mesh surface...")
    pcd = o3d_mesh.sample_points_uniformly(number_of_points=surface_samples)
    sampled_points = np.array(pcd.points)
    print(f"Surface sampling complete.")
    
    # Run the visibility-based selection
    print(f"\nStarting visibility-based skeleton point selection...")
    print(f"Parameters:")
    print(f"  - Initial candidates: {len(init_skeleton_points)}")
    print(f"  - Proximity threshold: {proximity_threshold}")
    print(f"  - Uniformity weight: {uniformity_weight}")
    print(f"  - Centrality weight: {centrality_weight}")
    print(f"  - Max points: {max_points if max_points else 'unlimited'}")
    
    selected_indices = _select_visibility_points(
        o3d_mesh,
        sampled_points,
        init_skeleton_points,
        proximity_threshold=proximity_threshold,
        uniformity_weight=uniformity_weight,
        centrality_weight=centrality_weight,
        max_points=max_points
    )
    
    # Get the final selected points
    selected_skeleton_points = init_skeleton_points[selected_indices]
    
    print("\n--- Selection Complete ---")
    print(f"Selected {len(selected_skeleton_points)} skeleton points from {len(init_skeleton_points)} candidates")
    
    return selected_skeleton_points, selected_indices


def _select_visibility_points(
    mesh: o3d.geometry.TriangleMesh,
    sampled_points: np.ndarray,
    init_skeleton_points: np.ndarray,
    proximity_threshold: float,
    uniformity_weight: float = 1.0,
    centrality_weight: float = 0.5,
    max_points: int = None
) -> np.ndarray:
    """
    Modified version of select_visibility_points2.
    
    Changes:
    - Retains the fast, explicit loop structure of v2.
    - Replaces the complex 'local clearance/symmetry' centrality with 
      the simple 'distance to surface' logic from v1.
    - Ensures uniformity is calculated based on distance to selected points.
    """
    
    # 1. Pre-compute the Coverage Matrix (Same as before)
    # This is the heavy lifting for visibility
    D = _compute_visibility_matrix(
        mesh, sampled_points, init_skeleton_points, proximity_threshold
    )

    m, n = D.shape
    
    # 2. Compute Centrality Scores (Simplified to match v1 logic)
    # V1 Logic: Centrality = Distance to the nearest surface point.
    print("Computing centrality scores (Distance to nearest surface point)...")
    
    # Calculate distances from all candidates to all surface samples
    # shape: (n_candidates, n_surface_samples)
    distances_to_surface = cdist(init_skeleton_points, sampled_points)
    
    # The score is simply the distance to the closest surface point (Medial Radius)
    centrality_scores = np.min(distances_to_surface, axis=1)
    
    # Normalize centrality scores once globally (optional, but helps with weighting)
    if np.std(centrality_scores) > 1e-6:
        centrality_scores_norm = (centrality_scores - np.mean(centrality_scores)) / np.std(centrality_scores)
    else:
        centrality_scores_norm = centrality_scores
    
    # Indices of surface points not yet covered
    S_uncovered = np.arange(m) 
    
    # Indices of selected candidate points
    A_selected_indices = []
    
    # Boolean mask of available candidate points
    P_available = np.ones(n, dtype=bool)

    if max_points is None:
        max_iter = n
    else:
        max_iter = min(max_points, n)

    for iteration in range(max_iter):
        
        # Stop if all surface points are covered
        if len(S_uncovered) == 0:
            print("All coverable points are covered.")
            break
            
        # --- 1. Coverage Score ---
        # Calculate score: how many *uncovered* points does each candidate cover?
        # Summing only the rows corresponding to uncovered surface points
        score = np.sum(D[S_uncovered], axis=0).astype(float)
        
        # Penalize already-selected or unavailable points immediately
        score[~P_available] = -np.inf
        
        # Stop if no available candidates can cover any new points
        if np.max(score) == 0:
            print("Stopping: No remaining candidates can cover new points.")
            break

        # Normalize Coverage Score
        # We only normalize based on the available points to keep the scale consistent
        valid_scores = score[P_available]
        score_std = np.std(valid_scores)
        if score_std > 1e-6:
            score_normalized = (score - np.mean(valid_scores)) / score_std
        else:
            score_normalized = score.copy()

        # --- 2. Centrality Score ---
        # Add centrality bonus (V1 logic: favoring interior points)
        if centrality_weight > 0:
            score_normalized += centrality_weight * centrality_scores_norm

        # --- 3. Uniformity Score ---
        # V1 Logic: Favor points far from ALREADY selected points
        if len(A_selected_indices) > 0 and uniformity_weight > 0:
            # Calculate distance from each candidate to the *nearest selected* candidate
            # This matches 'compute_min_distances' from CoverageAxis.py
            uniformity_dists = _compute_min_distances_to_subset(
                init_skeleton_points, init_skeleton_points[A_selected_indices]
            )
            
            # Normalize uniformity
            # Points far from existing skeleton get higher scores
            uni_valid = uniformity_dists[P_available]
            uni_std = np.std(uni_valid)
            
            if uni_std > 1e-6:
                uniformity_normalized = (uniformity_dists - np.mean(uni_valid)) / uni_std
                score_normalized += uniformity_weight * uniformity_normalized
        
        # Re-penalize unavailable points (to be safe after adding bonuses)
        score_normalized[~P_available] = -np.inf

        # --- 4. Selection ---
        i_k = np.argmax(score_normalized) # Find the best candidate

        # Stop if no valid point is found
        if not np.isfinite(score_normalized[i_k]):
            print("Stopping: No valid candidates left.")
            break

        # Add the best candidate to our selected set
        A_selected_indices.append(i_k)
        P_available[i_k] = False # Mark as unavailable

        # --- 5. Update Uncovered Set ---
        # Get the mask of newly covered points by this candidate
        covered_by_ik_mask = D[S_uncovered, i_k]
        
        # Remove these points from the uncovered set
        S_uncovered = S_uncovered[~covered_by_ik_mask]
        
        # Optional: Print progress every 10 points
        if iteration % 10 == 0:
            print(f"Selected point {i_k}. {len(S_uncovered)} surface points remaining.")

        # Optional: Print progress every 10 points
        if iteration % 10 == 0:
            current_uncovered_count = len(S_uncovered)
            total_points = len(sampled_points)
            total_covered_percentage = ((total_points - current_uncovered_count) / total_points) * 100
            
            if current_uncovered_count > 0:
                newly_covered_count = np.sum(covered_by_ik_mask)
                step_coverage_percentage = (newly_covered_count / current_uncovered_count) * 100
                print(f"Surfave points covered: {total_covered_percentage:.2f}%. {current_uncovered_count} surface points remaining.")
            else:
                print(f"Selected point {i_k}. All coverable points are now covered (100%).")
        

    return np.array(A_selected_indices)


def _compute_visibility_matrix(
    mesh: o3d.geometry.TriangleMesh,
    sampled_points: np.ndarray,
    init_skeleton_points: np.ndarray,
    proximity_threshold: float
) -> np.ndarray:
    """
    Computes the coverage matrix D based on proximity and visibility.

    Args:
        mesh: The 3D mesh object for raycasting
        sampled_points: (m, 3) array of surface points
        init_skeleton_points: (n, 3) array of candidate points
        proximity_threshold: The maximum distance to be considered "covered"

    Returns:
        (m, n) boolean matrix D, where D[j, i] is True if
        candidate i sees surface point j AND is within the threshold
    """
    m, n = len(sampled_points), len(init_skeleton_points)
    D = np.zeros((m, n), dtype=bool)

    # Epsilon to avoid self-intersection at the origin of the ray
    EPSILON = 1e-5 
    
    # Create a raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    
    # Convert mesh to tensor mesh and add to scene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    print(f"Computing visibility matrix for {m} surface points and {n} candidates...")
    
    for i in range(n):
        # 1. Get candidate point and setup rays
        p_i = init_skeleton_points[i]
        origins = np.tile(p_i, (m, 1))
        vectors = sampled_points - origins
        target_dists = np.linalg.norm(vectors, axis=1)
        
        # Avoid division by zero if a sample point is identical to a candidate
        safe_dists = target_dists.copy()
        safe_dists[safe_dists == 0] = 1e-9
        directions = vectors / safe_dists[:, np.newaxis]

        # 2. Visibility Check (for ALL points, not just nearby ones)
        # Perform ray intersections using Open3D
        rays = np.concatenate([
            origins.astype(np.float32),
            directions.astype(np.float32)
        ], axis=1)
        
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        result = scene.cast_rays(rays_tensor)
        
        # Get hit distances
        hit_dists = result['t_hit'].numpy()
        
        # A point is visible if the first hit distance is >= the target distance
        # (minus an epsilon to account for the ray hitting the target point itself)
        # hit_dists will be inf if no hit occurred
        is_visible = (hit_dists >= target_dists - EPSILON) | np.isinf(hit_dists)
        
        # 3. Proximity Check
        # A point must be within the proximity threshold
        is_close_enough = (target_dists <= proximity_threshold)
        
        # 4. Combine: A point is covered ONLY if it's both visible AND close enough
        D[:, i] = is_visible & is_close_enough
        
        if (i + 1) % (n // 10 or 1) == 0:
            print(f"  ...processed {i+1}/{n} candidates.")

    print("Visibility matrix computed.")
    return D

def generate_random_interior_points_raycasting(mesh, num_samples, min_distance_to_surface=0.0, max_iterations=100):
    
    # --- 1. PREPARE DATA ---
    if isinstance(mesh, (o3d.geometry.TriangleMesh, o3d.t.geometry.TriangleMesh)):
        if hasattr(mesh, "to_legacy"):
             mesh = mesh.to_legacy()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    elif isinstance(mesh, trimesh.Trimesh):
        tri_mesh = mesh
        vertices = mesh.vertices
        faces = mesh.faces
    else:
        raise ValueError("Input mesh must be Open3D TriangleMesh or Trimesh object.")

    min_bound = tri_mesh.bounds[0]
    max_bound = tri_mesh.bounds[1]

    # --- 2. INITIALIZE WARP SAMPLER (ONCE) ---
    # This triggers the 20s setup time exactly one time.
    sampler = WarpMeshSampler(vertices, faces)
    
    interior_points_list = []
    total_found = 0
    iteration = 0
    
    # Heuristic: Since GPU is fast, we can afford larger batches
    batch_size = max(num_samples * 5, 100000) 

    while total_found < num_samples and iteration < max_iterations:
        iteration += 1
        needed = num_samples - total_found

        # # Adaptive batch sizing
        # if iteration > 1 and len(interior_points_list) > 0:
        #     efficiency = total_found / (batch_size * (iteration - 1))
        #     if efficiency > 0:
        #         # If efficiency is low (e.g. 1%), we need huge batches
        #         current_batch_size = int(needed / efficiency * 1.5)
        #     else:
        #          current_batch_size = needed * 50
        # else:
        #      current_batch_size = batch_size
        
        # Cap batch size to fit in VRAM (e.g., 5 million points is safe on most GPUs)
        #current_batch_size = min(max(current_batch_size, 50000), 5000000)

        print(f"Iter {iteration}: Casting {batch_size} rays...")
        
        # A. Generate Random Points (CPU)
        points = np.random.uniform(min_bound, max_bound, size=(batch_size, 3))

        # B. GPU Parity Check (Keep only inside points)
        inside_mask = sampler.query_inside(points)
        batch_candidates = points[inside_mask]

        # C. GPU Distance Check (Replaces Trimesh!)
        # Only run if we have candidates and a distance constraint
        if len(batch_candidates) > 0 and min_distance_to_surface > 0.0:
            dist_mask = sampler.filter_too_close(batch_candidates, min_distance_to_surface)
            batch_candidates = batch_candidates[dist_mask]

        # Aggregate
        if len(batch_candidates) > 0:
            interior_points_list.append(batch_candidates)
            total_found += len(batch_candidates)

        print(f"Iter {iteration}: Found {len(batch_candidates)} valid. Total: {total_found}/{num_samples}")

    if total_found == 0:
        return np.empty((0, 3))

    interior_points = np.concatenate(interior_points_list, axis=0)
    if len(interior_points) > num_samples:
        indices = np.random.choice(len(interior_points), num_samples, replace=False)
        interior_points = interior_points[indices]

    return interior_points

def _compute_min_distances_to_subset(points: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """
    Helper function to compute the minimum distance from each point in 'points'
    to any point in 'subset'.
    
    Args:
        points: (n, 3) array of points
        subset: (m, 3) array of subset points
        
    Returns:
        (n,) array of minimum distances
    """
    from scipy.spatial.distance import cdist
    dists = cdist(points, subset)
    min_dists = np.min(dists, axis=1)
    return min_dists