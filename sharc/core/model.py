import os
import numpy as np
from dataclasses import dataclass
from sharc.math.sh import (
    fit_ref_points_SH, 
    smooth_coefficients_list, 
    pack_real_coeffs, 
    unpack_real_coeffs,
    s2fft_2d_to_dense,
    dense_to_s2fft_2d
)
from sharc.ops.reconstruction import filter_points_by_knn
from sharc.math.utils import estimate_outward_normals

@dataclass
class SharcConfig:

    L_fit: int = 64
    L_recon: int = 64
    N_fit: int = 1e+4
    N_recon: int = 2e+4
    lanczos_smoothing: bool = True
    k: int = 5


class SharcModel:
    def __init__(self, config, ref_points, coeffs):
        self.config = config
        self.ref_points = ref_points
        self.coeffs = coeffs
        

    def fit(self, mesh, device='cuda'):
        coeffs = fit_ref_points_SH(
            mesh,
            self.ref_points,
            self.config.N_fit,
            self.config.L_fit,
            device=device
        )
        if self.config.lanczos_smoothing:
            coeffs = smooth_coefficients_list(coeffs, self.config.L_fit)
        self.coeffs = coeffs
        return coeffs

    def reconstruct(self):
        filtered_points, filtered_indices = filter_points_by_knn(
            self.ref_points,
            self.coeffs,
            self.config.N_recon,
            self.config.L_recon,
            k=self.config.k
        )
        normals = estimate_outward_normals(filtered_points, self.ref_points[filtered_indices])
        return filtered_points, normals

    @classmethod
    def load(cls, path):
        """
        Loads SHARC model. 
        Promotes data back to complex128 for compatibility with Healpy/JAX.
        """
        if os.path.isdir(path):
            file_path = os.path.join(path, 'sh_coeffs.npz')
        else:
            file_path = path
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model not found: {file_path}")

        print(f"Loading SHARC model from {file_path}...")
        data = np.load(file_path)
        
        # Load Anchors
        anchors = data['origins']
        
        # Load Lmax
        lmax = int(data['lmax']) if 'lmax' in data else 64 
        raw_coeffs = data['coeffs']

        # Unpack Symmetry
        if 'is_packed' in data and data['is_packed']:
            # Safety check on size
            K = raw_coeffs.shape[-1]
            M = (-1 + np.sqrt(1 + 8 * K)) / 2
            inferred_lmax = int(M)
            if inferred_lmax != lmax:
                lmax = inferred_lmax
            
            coeffs_dense = unpack_real_coeffs(raw_coeffs, lmax)
        else:
            coeffs_dense = raw_coeffs

        # Healpy crashes if we feed it complex64. 
        # We save as complex64 for disk space, but run as complex128.
        coeffs_dense = coeffs_dense.astype(np.complex128)

        # Convert 1D Dense -> 2D Padded
        coeffs_2d_list = [dense_to_s2fft_2d(c, lmax) for c in coeffs_dense]
        
        # Reconstruct Config & Model
        default_config = SharcConfig(L_fit=lmax)
        print(f"  Model loaded: {len(anchors)} anchors, L={lmax}")
        
        return cls(config=default_config, ref_points=anchors, coeffs=coeffs_2d_list)

    def save(self, path):
        # Standardize and Convert 2D -> 1D Dense
        if isinstance(self.coeffs, list):
            # Convert each anchor's 2D array to a 1D dense array
            coeffs_1d_list = [s2fft_2d_to_dense(np.array(c)) for c in self.coeffs]
            coeffs_np = np.stack(coeffs_1d_list)
        else:
            # Assume it's already a stack of 2D arrays (N, L, 2L-1)
            # We map the function over the first axis
            coeffs_np = np.array([s2fft_2d_to_dense(c) for c in self.coeffs])
            
        anchors_np = np.array(self.ref_points)

        # Cast to Single Precision
        anchors_f32 = anchors_np.astype(np.float32)
        coeffs_c64 = coeffs_np.astype(np.complex64)

        # Spatial Sorting
        sort_order = np.lexsort((anchors_f32[:, 2], anchors_f32[:, 1], anchors_f32[:, 0]))
        anchors_sorted = anchors_f32[sort_order]
        coeffs_sorted = coeffs_c64[sort_order]

        # Symmetry Packing
        # Because are signal is a distance field which is real, keeping both +m and -m coefficients is redundant,
        # we can pack the coefficients to save space.
        coeffs_packed = pack_real_coeffs(coeffs_sorted, self.config.L_fit)

        # Save
        save_path = os.path.join(path, 'sh_coeffs.npz')
        np.savez(
            save_path, 
            coeffs=coeffs_packed, 
            origins=anchors_sorted, 
            is_packed=True,
            lmax=self.config.L_fit
        )
        print(f"Saved optimized model to: {save_path}")