import os
import argparse
import time
import numpy as np
import open3d as o3d
import jax
from dataclasses import dataclass

# Enable x64 BEFORE any JAX ops are run
jax.config.update("jax_enable_x64", True)

# Import SHARC Core
from sharc.core.model import SharcModel, SharcConfig

def run_reconstruction(args):
    print(f"Loading model from: {args.model_path}")
    start_time = time.time()
    
    # 1. Load Model
    # The load method automatically handles unpacking compressed coefficients
    model = SharcModel.load(args.model_path)
    
    # Update config for reconstruction
    # We explicitly overwrite the config with command line args, 
    # ensuring the model uses the requested reconstruction parameters.
    model.config.L_recon = args.lrecon
    model.config.N_recon = args.N_recon
    model.config.k = args.k_nn
    
    print(f"  Model loaded with {len(model.ref_points)} reference points.")

    # 2. Reconstruct
    print(f"Reconstructing with L={args.lrecon}, N={args.N_recon}...")
    recon_points, recon_normals = model.reconstruct()
    
    print(f"  Generated {len(recon_points)} points.")

    # 3. Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save Point Cloud
    pcd_path = os.path.join(args.output_dir, 'reconstructed_pcd.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(recon_points)
    pcd.normals = o3d.utility.Vector3dVector(recon_normals)
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"  Point cloud saved: {pcd_path}")

    # 4. Poisson Meshing (Optional)
    if args.mesh:
        print("Running Poisson meshing...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth
        )
        mesh_path = os.path.join(args.output_dir, 'reconstructed_mesh.ply')
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"  Mesh saved: {mesh_path}")

    print(f"Done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHARC Reconstruction from Saved Model")
    
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model directory or sh_coeffs.npz file')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save results')
    
    # Reconstruction params
    parser.add_argument('--lrecon', type=int, default=64, help='SH degree for reconstruction')
    parser.add_argument('--N_recon', type=int, default=100000, help='Number of candidate points for reconstruction')
    parser.add_argument('--k_nn', type=int, default=5, help='K-NN for filtering')
    
    # Meshing
    parser.add_argument('--mesh', action='store_true', help='Perform Poisson meshing')
    parser.add_argument('--poisson_depth', type=int, default=9, help='Depth for Poisson reconstruction')
    
    args = parser.parse_args()
    run_reconstruction(args)
