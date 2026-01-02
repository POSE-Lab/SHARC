import warp as wp
import numpy as np

# --- THE HELPER CLASS ---
class WarpMeshSampler:
    def __init__(self, vertices, faces):
        self.device = 'cuda'
        print("Uploading mesh to GPU and building BVH (Once)...")
        self.wp_points = wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.wp_indices = wp.array(faces.astype(np.int32).flatten(), dtype=int, device=self.device)
        self.mesh = wp.Mesh(points=self.wp_points, indices=self.wp_indices, velocities=None)
        print("BVH Ready.")

    def query_inside(self, points_np):
        """Parity Check: Returns mask of points inside the mesh."""
        n = len(points_np)
        if n == 0: return np.array([], dtype=bool)
        
        points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=self.device)
        mask_wp = wp.zeros(n, dtype=int, device=self.device)
        
        wp.launch(
            kernel=robust_parity_check_kernel,
            dim=n,
            inputs=[self.mesh.id, points_wp, mask_wp, 1e-4],
            device=self.device
        )
        return mask_wp.numpy().astype(bool)

    def filter_too_close(self, points_np, min_dist):
        """Distance Check: Returns mask of points that are FAR ENOUGH from surface."""
        n = len(points_np)
        if n == 0: return np.array([], dtype=bool)
        if min_dist <= 0: return np.ones(n, dtype=bool)

        points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=self.device)
        mask_wp = wp.zeros(n, dtype=int, device=self.device)

        # Launch the distance kernel
        wp.launch(
            kernel=distance_filter_kernel,
            dim=n,
            inputs=[self.mesh.id, points_wp, mask_wp, min_dist],
            device=self.device
        )
        
        return mask_wp.numpy().astype(bool)

def launch_ray_cast(wp_mesh,
                    origin,
                    directions_wp,
                    distances_wp,
                    num_samples,
                    max_distance=2.0,
                    device='cuda'):
    
    origin_np = np.array(origin, dtype=np.float32)
    origins_repeated = np.tile(origin_np, (num_samples, 1))
    origins_wp = wp.array(origins_repeated, dtype=wp.vec3, device=device)

    # Launch the optimized kernel
    wp.launch(
        kernel=optimized_ray_cast_kernel,
        dim=num_samples,
        inputs=[wp_mesh.id, origins_wp, directions_wp, distances_wp, max_distance],
        device=device
    )
    wp.synchronize()
    distances = distances_wp.numpy()
    
    # Filter valid hits
    mask = (distances > 0) & (distances < max_distance) & np.isfinite(distances)
    final_distances = np.zeros_like(distances)
    final_distances[mask] = distances[mask]
    
    return final_distances

@wp.kernel
def distance_filter_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    valid_mask: wp.array(dtype=int),
    min_dist: float
):
    i = wp.tid()
    p = points[i]
    
    # "Is there any triangle within min_dist of this point?"
    # If yes (result=True), the point is too close -> Invalid (0)
    # If no (result=False), the point is safe -> Valid (1)
    
    query = wp.mesh_query_point(mesh_id, p, min_dist)
    
    if query.result:
        valid_mask[i] = 0 # Too close
    else:
        valid_mask[i] = 1 # Far enough

@wp.kernel
def robust_parity_check_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    inside_mask: wp.array(dtype=int),
    epsilon: float
):
    i = wp.tid()
    p = points[i]
    
    votes = int(0)
    
    # 3 Orthogonal Rays
    dirs = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )

    for d in range(3):
        direction = dirs[d]
        t_current = float(0.0)
        hit_count = int(0)
        
        # Max 100 bounces to prevent infinite loops
        for safe in range(100): 
            query = wp.mesh_query_ray(mesh_id, p + direction * t_current, direction, 1.0e6)
            
            if not query.result:
                break
            
            hit_count += 1
            t_current += query.t + epsilon
        
        if hit_count % 2 == 1:
            votes += 1

    if votes >= 2:
        inside_mask[i] = 1
    else:
        inside_mask[i] = 0
        
@wp.kernel
def optimized_ray_cast_kernel(
    mesh_id: wp.uint64,
    ray_origins: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    hit_distances: wp.array(dtype=float),
    max_dist: float
):
    i = wp.tid()
    
    # wp.mesh_query_ray returns true if hit, plus t, u, v, sign, normal, face_index
    query = wp.mesh_query_ray(mesh_id, ray_origins[i], ray_directions[i], max_dist)

    if query.result:
        hit_distances[i] = query.t
    else:
        hit_distances[i] = -1.0