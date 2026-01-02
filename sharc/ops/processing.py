import os
import sys
import glob
import subprocess
import traceback
import numpy as np
import open3d as o3d
import trimesh
import pymeshlab
from typing import Tuple, Optional, Dict
from contextlib import contextmanager
from sharc.math.transforms import fitModel2UnitSphere, move_origin_to_lower_left, calc_max_dist

class MeshPreprocessor:
    """
    Handles mesh preprocessing including loading, simplification, normalization, and repair.
    
    Attributes:
        original_mesh: Original input mesh
        simplified_mesh: Simplified mesh
        normalized_mesh_simplified: Normalized simplified mesh
        normalized_mesh_detailed: Normalized detailed mesh
        scale: Normalization scale factor
        max_distance: Maximum distance used for normalization
        sphere_center: Center of the minimal enclosing sphere
        centroid: Centroid used for centering
        transformation_params: Dictionary storing transformation parameters
    """
    
    def __init__(self, mesh_path: Optional[str] = None):
        """
        Initialize MeshPreprocessor.
        
        Args:
            mesh_path: Path to input mesh file (optional)
        """
        self.original_mesh = None
        self.simplified_mesh = None
        self.normalized_mesh_simplified = None
        self.normalized_mesh_detailed = None
        self.scale = None
        self.max_distance = None
        self.sphere_center = None
        self.centroid = None
        self.transformation_params = {}
        
        if mesh_path:
            self.load_mesh(mesh_path)
    
    def load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """
        Load mesh from file.
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Loaded trimesh object
        """
        print(f"Loading mesh from: {mesh_path}")
        self.original_mesh = trimesh.load(mesh_path)
        print(f"  Vertices: {len(self.original_mesh.vertices)}, Faces: {len(self.original_mesh.faces)}")
        return self.original_mesh
    
    def simplify(self, target_num_faces: int, method: str = 'quadric') -> trimesh.Trimesh:
        """
        Simplify mesh using PyMeshLab.
        
        Args:
            target_num_faces: Target number of faces after simplification
            method: Simplification method ('quadric' or 'clustering')
            
        Returns:
            Simplified trimesh object
        """
        if self.original_mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
        
        print(f"Simplifying mesh to {target_num_faces} faces using {method} method...")
        
        # Suppress pymeshlab's verbose output (plugin loading messages)
        with suppress_pymeshlab_output():
            ms = pymeshlab.MeshSet()
            
            vertices = np.asarray(self.original_mesh.vertices)
            faces = np.asarray(self.original_mesh.faces)
            
            m = pymeshlab.Mesh(vertices, faces)
            ms.add_mesh(m)
            
            self._repair_mesh(ms)
            
            if method == 'quadric':
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces,  preserveboundary = True)
            elif method == 'clustering':
                ms.meshing_decimation_clustering(threshold=pymeshlab.Percentage(50))
            else:
                raise ValueError(f"Unknown simplification method: {method}")
            
            self._repair_mesh(ms)
        
        simplified_mesh_pml = ms.current_mesh()
        vertices = simplified_mesh_pml.vertex_matrix()
        faces = simplified_mesh_pml.face_matrix()
        
        self.simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f"  Simplified mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        return self.simplified_mesh
    
    def _repair_mesh(self, meshset: pymeshlab.MeshSet) -> None:
        """
        Apply mesh repair operations.
        
        Args:
            meshset: PyMeshLab MeshSet object
        """
        meshset.meshing_remove_duplicate_vertices()
        meshset.meshing_remove_duplicate_faces()
        meshset.meshing_repair_non_manifold_edges()
        meshset.meshing_remove_null_faces()
        meshset.meshing_repair_non_manifold_vertices(vertdispratio=0)
        meshset.meshing_repair_non_manifold_edges(method=0)
        meshset.meshing_remove_unreferenced_vertices()
    
    def repair_active_mesh(self, mesh_type: str = 'normalized_simplified') -> trimesh.Trimesh:
        """
        Explicitly repair a specific mesh (in-place).
        
        Args:
            mesh_type: Which mesh to repair ('simplified', 'normalized_simplified', 'original', 'normalized_detailed')
            
        Returns:
            Repaired trimesh object
        """
        print(f"Repairing mesh: {mesh_type}...")
        
        target_mesh = None
        if mesh_type == 'simplified':
            target_mesh = self.simplified_mesh
        elif mesh_type == 'normalized_simplified':
            target_mesh = self.normalized_mesh_simplified
        elif mesh_type == 'original':
            target_mesh = self.original_mesh
        elif mesh_type == 'normalized_detailed':
            target_mesh = self.normalized_mesh_detailed
            
        if target_mesh is None:
             raise ValueError(f"Mesh type {mesh_type} is not available (None).")
             
        with suppress_pymeshlab_output():
            ms = pymeshlab.MeshSet()
            vertices = np.asarray(target_mesh.vertices)
            faces = np.asarray(target_mesh.faces)
            m = pymeshlab.Mesh(vertices, faces)
            ms.add_mesh(m)
            
            self._repair_mesh(ms)
            
            repaired_mesh_pml = ms.current_mesh()
            vertices = repaired_mesh_pml.vertex_matrix()
            faces = repaired_mesh_pml.face_matrix()
            
            new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
        print(f"  Repaired mesh stats: {len(vertices)} vertices, {len(faces)} faces")
        
        if mesh_type == 'simplified':
            self.simplified_mesh = new_mesh
        elif mesh_type == 'normalized_simplified':
            self.normalized_mesh_simplified = new_mesh
        elif mesh_type == 'original':
            self.original_mesh = new_mesh
        elif mesh_type == 'normalized_detailed':
            self.normalized_mesh_detailed = new_mesh
            
        return new_mesh
    
    def normalize(self, 
                 buffer: float = 1.0, 
                 scale: float = 1.0,
                 move_to_origin: bool = True,
                 use_simplified: bool = True) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        Normalize mesh to unit sphere and optionally move to origin.
        
        Creates two normalized meshes:
        - Simplified: For skeleton computation (QMAT, medial axis)
        - Detailed: For reconstruction (preserves surface details)
        
        Args:
            buffer: Buffer factor for sphere fitting (>1 expands sphere)
            scale: Final scale factor
            move_to_origin: Move bounding box minimum to origin
            use_simplified: Use simplified mesh for computing normalization params
            
        Returns:
            Tuple of (normalized_simplified, normalized_detailed)
        """
        if self.original_mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
        
        print("Normalizing mesh...")
        
        if use_simplified and self.simplified_mesh is not None:
            reference_verts = np.array(self.simplified_mesh.vertices)
            print("  Using simplified mesh for normalization parameters")
        else:
            reference_verts = np.array(self.original_mesh.vertices)
            print("  Using original mesh for normalization parameters")
        
        norm_verts_ref, self.scale, self.max_distance, sphere_center = fitModel2UnitSphere(
            reference_verts.copy(), buffer=buffer, scale=scale
        )
        self.sphere_center = sphere_center
        
        # Calculate effective radius used in fitModel2UnitSphere
        # effective_radius = radius / buffer
        # We can recover it from the return values: 
        # The function returns: points_normalized, scale, effective_radius, center
        # So self.max_distance is actually the effective_radius (based on how fitModel2UnitSphere is implemented in transforms.py)
        # Wait, let's check transforms.py again.
        # fitModel2UnitSphere returns: points_normalized, scale, effective_radius, center
        # So the 3rd return value IS effective_radius.
        
        effective_radius = self.max_distance
        
        print(f"  Scale factor: {self.scale:.6f}")
        print(f"  Effective radius: {effective_radius:.6f}")
        print(f"  Sphere center: {self.sphere_center}")
        print(f"  Normalized bounds: min={norm_verts_ref.min(axis=0)}, max={norm_verts_ref.max(axis=0)}")
        
        # Define normalization function to ensure consistency
        def apply_normalization(verts, center, eff_radius, s):
            return (verts - center) / eff_radius * s

        if self.simplified_mesh is not None:
            simplified_verts = np.array(self.simplified_mesh.vertices)
            # Apply SAME normalization
            norm_verts_simplified = apply_normalization(
                simplified_verts, self.sphere_center, effective_radius, self.scale
            )
            
            self.normalized_mesh_simplified = trimesh.Trimesh(
                vertices=norm_verts_simplified,
                faces=self.simplified_mesh.faces
            )
        
        original_verts = np.array(self.original_mesh.vertices)
        # Apply SAME normalization
        norm_verts_detailed = apply_normalization(
            original_verts, self.sphere_center, effective_radius, self.scale
        )
        
        self.normalized_mesh_detailed = trimesh.Trimesh(
            vertices=norm_verts_detailed,
            faces=self.original_mesh.faces
        )
        
        print(f"  Detailed mesh preserved: {len(norm_verts_detailed)} vertices")
        
        self.transformation_params = {
            'buffer': buffer,
            'scale': scale,
            'move_to_origin': move_to_origin,
            'normalization_scale': self.scale,
            'max_distance': self.max_distance, # This is effective_radius
            'sphere_center': self.sphere_center.tolist()
        }
        
        return self.normalized_mesh_simplified, self.normalized_mesh_detailed
    
    def center_mesh(self, 
                    mesh: Optional[trimesh.Trimesh] = None,
                    centroid: Optional[np.ndarray] = None) -> trimesh.Trimesh:
        """
        Center mesh around a given centroid.
        
        Args:
            mesh: Mesh to center (uses normalized_mesh_simplified if None)
            centroid: Centroid to use (computes from mesh if None)
            
        Returns:
            Centered trimesh object
        """
        if mesh is None:
            if self.normalized_mesh_simplified is None:
                raise ValueError("No normalized mesh available")
            mesh = self.normalized_mesh_simplified
        
        vertices = np.array(mesh.vertices)
        
        if centroid is None:
            centroid = np.mean(vertices, axis=0)
        
        self.centroid = centroid
        centered_vertices = vertices - centroid
        
        centered_mesh = trimesh.Trimesh(vertices=centered_vertices, faces=mesh.faces)
        
        print(f"Mesh centered around: {centroid}")
        
        return centered_mesh
    
    def compute_medial_axis(self,
                           qmat_exec_path: str,
                           output_dir: str,
                           mesh: Optional[trimesh.Trimesh] = None) -> str:
        """
        Compute medial axis using QMAT mesh_to_ma tool.
        
        Args:
            qmat_exec_path: Path to QMAT executable directory
            output_dir: Output directory for medial axis file
            mesh: Mesh to use (uses normalized_mesh_simplified if None)
            
        Returns:
            Path to generated medial axis file
        """
        if mesh is None:
            if self.normalized_mesh_detailed is None:
                raise ValueError("No normalized mesh available")
            mesh = self.normalized_mesh_detailed
        
        os.makedirs(output_dir, exist_ok=True)
        
        input_off = os.path.join(output_dir, "input.off")
        mesh.export(input_off)
        
        medial_axis_file = os.path.join(output_dir)
        mesh_to_ma_cmd = [
            os.path.join(qmat_exec_path, "mesh_to_ma"),
            input_off,
            medial_axis_file
        ]
        
        print(f"Computing medial axis...")
        result = subprocess.run(mesh_to_ma_cmd, capture_output=True, text=True)

        
        
        if result.returncode != 0:
            raise RuntimeError(f"mesh_to_ma failed: {result.stderr}")
        
        print(f"  Medial axis saved: {medial_axis_file}")
        
        return medial_axis_file
    
    def _filter_ma_edges(self, ma_file_path: str, mesh: trimesh.Trimesh) -> None:
        """
        Filter MA edges that intersect with the mesh and update the file.
        
        Args:
            ma_file_path: Path to the .ma file
            mesh: Trimesh object used for intersection checking
        """
        print(f"  Filtering edges in medial axis: {ma_file_path}")
        with open(ma_file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return
            
        header = lines[0].strip().split()
        if len(header) != 3:
            print("  Warning: Invalid MA header, skipping filter.")
            return
            
        n_v, n_e, n_f = map(int, header)
        
        # Read vertices
        v_lines = lines[1:1+n_v]
        vertices = []
        for line in v_lines:
            parts = line.strip().split()
            # v x y z r
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        vertices = np.array(vertices, dtype=np.float64)
        
        # Read edges
        e_lines = lines[1+n_v : 1+n_v+n_e]
        edges = []
        for line in e_lines:
            parts = line.strip().split()
            # e v1 v2
            edges.append([int(parts[1]), int(parts[2])])
        edges = np.array(edges, dtype=np.int64) # Ensure integer type
        
        if len(edges) == 0:
            return

        # Prepare rays for edges
        if edges.dtype != np.int64 and edges.dtype != np.int32:
             print(f"  DEBUG: edges dtype is {edges.dtype}, casting to int64")
             edges = edges.astype(np.int64)
             
        # Check indices bounds
        if edges.max() >= len(vertices):
            print(f"  Warning: Edge indices out of bounds. Max index: {edges.max()}, Vertices: {len(vertices)}")
            return

        origins = vertices[edges[:, 0]]
        targets = vertices[edges[:, 1]]
        vectors = targets - origins
        lengths = np.linalg.norm(vectors, axis=1)
        
        # Filter zero-length edges
        valid_b = lengths > 1e-6
        if not np.any(valid_b):
            return
            
        origins = origins[valid_b]
        vectors = vectors[valid_b]
        lengths = lengths[valid_b]
        original_indices = np.where(valid_b)[0]
        
        directions = vectors / lengths[:, np.newaxis]
        
        print(f"  DEBUG: Checking intersections for {len(origins)} valid rays.")
        
        # Check intersections
        # Using mesh.ray.intersects_location
        # Returns: locations, index_ray, index_tri
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=True
        )
        
        edges_to_remove = set()
        if len(index_ray) > 0:
            # Check indices types just in case
            if index_ray.dtype != np.int64 and index_ray.dtype != np.int32:
                 index_ray = index_ray.astype(np.int64)

            hit_origins = origins[index_ray]
            dists = np.linalg.norm(locations - hit_origins, axis=1)
            
            # Epsilon for robust check
            epsilon = 1e-4
            
            for r_idx, d in zip(index_ray, dists):
                l = lengths[r_idx]
                if d > epsilon and d < l - epsilon:
                    original_idx = original_indices[r_idx]
                    edges_to_remove.add(int(original_idx)) # Explicitly cast to Python int
            
            for r_idx, d in zip(index_ray, dists):
                l = lengths[r_idx]
                if d > epsilon and d < l - epsilon:
                    original_idx = original_indices[r_idx]
                    edges_to_remove.add(int(original_idx)) # Explicitly cast to Python int
        
        if not edges_to_remove:
            print("  No intersecting edges found.")
            return
            
        print(f"  Removing {len(edges_to_remove)} edges intersecting the mesh.")
        
        # Filter edges
        keep_mask = np.ones(len(edges), dtype=bool)
        if edges_to_remove:
            # Convert set to list of integers for indexing
            remove_indices = list(edges_to_remove)
            keep_mask[remove_indices] = False
            
        new_edges = edges[keep_mask]
        
        # Keep all vertices and faces (as requested)
        # We don't prune unused vertices anymore.
        
        # Faces lines
        f_lines = lines[1+n_v+n_e : 1+n_v+n_e+n_f]
        
        # Construct new edges lines using original indices
        new_e_lines = []
        for e in new_edges:
            new_e_lines.append(f"e {e[0]} {e[1]}\n")
            
        new_n_e = len(new_e_lines)
        
        with open(ma_file_path, 'w') as f:
            # Header: n_v same, n_e updated, n_f same
            f.write(f"{n_v} {new_n_e} {n_f}\n")
            f.writelines(v_lines)
            f.writelines(new_e_lines)
            f.writelines(f_lines)
            
        print(f"  Updated MA file stats: {n_v} vertices (kept all), {n_e}->{new_n_e} edges")


    
    def save_meshes(self, output_dir: str, save_all: bool = True) -> Dict[str, str]:
        """
        Save meshes to files.
        
        Args:
            output_dir: Output directory
            save_all: If True, save all available meshes
            
        Returns:
            Dictionary mapping mesh type to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        if self.original_mesh is not None:
            path = os.path.join(output_dir, 'original.ply')
            self.original_mesh.export(path)
            saved_paths['original'] = path
            print(f"Saved original mesh: {path}")
        
        if save_all and self.simplified_mesh is not None:
            path = os.path.join(output_dir, 'simplified.ply')
            self.simplified_mesh.export(path)
            saved_paths['simplified'] = path
            print(f"Saved simplified mesh: {path}")
        
        if save_all and self.normalized_mesh_simplified is not None:
            self.normalized_mesh_simplified.fix_normals()
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(
                np.array(self.normalized_mesh_simplified.vertices)
            )
            mesh_o3d.triangles = o3d.utility.Vector3iVector(
                np.array(self.normalized_mesh_simplified.faces)
            )
            mesh_o3d.compute_vertex_normals()
            
            path = os.path.join(output_dir, 'normalized_simplified.ply')
            o3d.io.write_triangle_mesh(path, mesh_o3d, write_vertex_normals=True)
            saved_paths['normalized_simplified'] = path
            print(f"Saved normalized simplified mesh: {path}")
        
        if save_all and self.normalized_mesh_detailed is not None:
            path = os.path.join(output_dir, 'normalized_detailed.ply')
            self.normalized_mesh_detailed.export(path)
            saved_paths['normalized_detailed'] = path
            print(f"Saved normalized detailed mesh: {path}")
        
        return saved_paths
    
    def export_to_off(self, output_path: str, mesh: Optional[trimesh.Trimesh] = None) -> str:
        """
        Export mesh to OFF format (required for QMAT).
        
        Args:
            output_path: Output file path
            mesh: Mesh to export (uses normalized_mesh_simplified if None)
            
        Returns:
            Path to exported file
        """
        if mesh is None:
            if self.normalized_mesh_simplified is not None:
                mesh = self.normalized_mesh_simplified
                print("  Exporting normalized_simplified mesh to OFF.")
            elif self.normalized_mesh_detailed is not None:
                mesh = self.normalized_mesh_detailed
                print("  Exporting normalized_detailed mesh to OFF (fallback).")
            else:
                raise ValueError("No normalized mesh available for export.")
        
        mesh.export(output_path)
        print(f"Exported mesh to OFF format: {output_path}")
        
        return output_path
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about processed meshes.
        
        Returns:
            Dictionary with mesh statistics
        """
        stats = {}
        
        if self.original_mesh is not None:
            stats['original'] = {
                'num_vertices': len(self.original_mesh.vertices),
                'num_faces': len(self.original_mesh.faces),
                'bounds': self.original_mesh.bounds.tolist()
            }
        
        if self.simplified_mesh is not None:
            stats['simplified'] = {
                'num_vertices': len(self.simplified_mesh.vertices),
                'num_faces': len(self.simplified_mesh.faces)
            }
        
        if self.normalized_mesh_simplified is not None:
            stats['normalized_simplified'] = {
                'num_vertices': len(self.normalized_mesh_simplified.vertices),
                'num_faces': len(self.normalized_mesh_simplified.faces),
                'bounds': self.normalized_mesh_simplified.bounds.tolist()
            }
        
        if self.normalized_mesh_detailed is not None:
            stats['normalized_detailed'] = {
                'num_vertices': len(self.normalized_mesh_detailed.vertices),
                'num_faces': len(self.normalized_mesh_detailed.faces)
            }
        
        if self.transformation_params:
            stats['transformation'] = self.transformation_params
        
        if self.centroid is not None:
            stats['centroid'] = self.centroid.tolist()
        
        return stats
    
    def apply_full_preprocessing(self,
                                target_num_faces: int = 20000,
                                buffer: float = 1.0,
                                scale: float = 1.0,
                                move_to_origin: bool = True,
                                simplification_method: str = 'quadric') -> Dict[str, trimesh.Trimesh]:
        """
        Apply full preprocessing pipeline.
        
        Args:
            target_num_faces: Target faces for simplification
            buffer: Buffer for normalization
            scale: Scale for normalization
            move_to_origin: Move to origin flag
            simplification_method: Method for simplification
            
        Returns:
            Dictionary with all processed meshes
        """
        print("="*60)
        print("Starting Full Mesh Preprocessing Pipeline")
        print("="*60)
        
        if self.original_mesh is None:
            raise ValueError("No mesh loaded. Call load_mesh() first.")
        
        self.simplify(target_num_faces, method=simplification_method)
        self.normalize(buffer=buffer, scale=scale, move_to_origin=move_to_origin)
        
        print("\n" + "="*60)
        print("Preprocessing Complete")
        print("="*60)
        
        return {
            'original': self.original_mesh,
            'simplified': self.simplified_mesh,
            'normalized_simplified': self.normalized_mesh_simplified,
            'normalized_detailed': self.normalized_mesh_detailed
        }
    
    def __repr__(self) -> str:
        """String representation."""
        parts = []
        if self.original_mesh:
            parts.append(f"original={len(self.original_mesh.vertices)}v")
        if self.simplified_mesh:
            parts.append(f"simplified={len(self.simplified_mesh.vertices)}v")
        if self.normalized_mesh_simplified:
            parts.append("normalized=True")
        
        return f"MeshPreprocessor({', '.join(parts)})"