import os
import argparse
import time
import jax
import json
import numpy as np
import open3d as o3d
from dataclasses import asdict

jax.config.update("jax_enable_x64", True)

# Import internal utils
from sharc.ops.processing import MeshPreprocessor
from sharc.ops.selection import (
    compute_skeleton_points_visibility, 
    generate_random_interior_points_raycasting
)
from sharc.utils.conversions import trimesh_to_open3d
from sharc.utils.io import create_output_structure

# Import SHARC Core
from sharc.core.model import SharcModel, SharcConfig

def run_pipeline(args):
    start_time = time.time()
    
    # Setup Output Structure
    paths = create_output_structure(args.input_mesh, args.output_dir)
    print(f"Output directory: {paths['root']}")

    # Preprocessing (Normalization)
    print("[1/5] Preprocessing Mesh...")
    preprocessor = MeshPreprocessor(args.input_mesh)
    preprocessor.normalize(buffer=1.0, scale=1.0, move_to_origin=True, use_simplified=False)
    preprocessor.save_meshes(paths['input'])
    mesh_detailed = preprocessor.normalized_mesh_detailed
    
    # Reference Points Selection
    print("[2/5] Generating Reference Points...")
    mesh_o3d = trimesh_to_open3d(mesh_detailed)
    
    # Initial random sampling
    random_points = generate_random_interior_points_raycasting(
        mesh_o3d,
        num_samples=args.random_samples,
        min_distance_to_surface=args.min_distance_random
    )
    
    # Optimal reference point selection
    ref_points, _ = compute_skeleton_points_visibility(
        mesh_detailed,
        surface_samples=args.num_surface_samples,
        init_skeleton_points=random_points,
        proximity_threshold=args.proximity_threshold,
        uniformity_weight=args.uniformity_weight,
        centrality_weight=args.centrality_weight,
        min_distance_from_surface=args.min_distance_from_surface,
        max_points=args.max_reference_points
    )
    print(f"  Selected {len(ref_points)} reference points.")
    
    # Save skeleton
    skeleton_pcd = o3d.geometry.PointCloud()
    skeleton_pcd.points = o3d.utility.Vector3dVector(ref_points)
    o3d.io.write_point_cloud(os.path.join(paths['ref_points'], 'skeleton.ply'), skeleton_pcd)

    # 4. Initialize and Fit SHARC Model
    print("[3/5] Fitting SHARC Model...")
    
    # Create Config from Args
    config = SharcConfig(
        L_fit=args.lfit,
        L_recon=args.lrecon,
        N_fit=args.N_coarse,
        N_recon=args.N_recon,
        lanczos_smoothing=args.lanczos_smoothing,
        k=args.k_nn
    )
    
    model = SharcModel(config, ref_points=ref_points, coeffs=None)
    repr_time = time.time()
    model.fit(mesh_o3d, device=args.device)
    repr_time = time.time() - repr_time
    
    # Save Model
    model.save(paths['coeffs'])

    # Metadata
    metadata = {
        'input': args.input_mesh,
        'config': asdict(config),
        'metrics': {
            'num_ref_points': len(ref_points),
            'total_time': time.time() - start_time,
            'repr_time': repr_time
        }
    }
    with open(os.path.join(paths['root'], 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Pipeline Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHARC Pipeline Entrypoint")
    
    # Input/Output
    parser.add_argument('--input_mesh', type=str, required=True, help='Path to input mesh')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    
    # Preprocessing / Skeletonization
    parser.add_argument('--random_samples', type=int, default=10000)
    parser.add_argument('--min_distance_random', type=float, default=0.001)
    parser.add_argument('--num_surface_samples', type=int, default=10000)
    parser.add_argument('--proximity_threshold', type=float, default=0.2)
    parser.add_argument('--uniformity_weight', type=float, default=1.0)
    parser.add_argument('--centrality_weight', type=float, default=1.0)
    parser.add_argument('--min_distance_from_surface', type=float, default=1e-4)
    parser.add_argument('--max_reference_points', type=int, default=1000)
    
    # SHARC Model Params
    parser.add_argument('--lfit', type=int, default=64, help='SH degree for fitting')
    parser.add_argument('--lrecon', type=int, default=64, help='SH degree for reconstruction')
    parser.add_argument('--N_coarse', type=int, default=10000, help='Fitting samples (rays)')
    parser.add_argument('--N_recon', type=int, default=20000, help='Reconstruction candidates')
    parser.add_argument('--lanczos_smoothing', type=bool, default=True, help='Apply Lanczos smoothing to SH coefficients')
    parser.add_argument('--k_nn', type=int, default=5, help='K-NN for reconstruction filtering')
    
    # System
    parser.add_argument('--device', type=str, default='cuda', help='Compute device')

    args = parser.parse_args()
    run_pipeline(args)
