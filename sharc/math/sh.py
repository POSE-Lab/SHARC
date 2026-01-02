import numpy as np
import s2fft
import jax
import warp as wp
import healpy as hp
import tqdm
from sharc.utils.warp_utils import launch_ray_cast
from sharc.math.transforms import spherical_to_cartesian

METHOD = "jax_healpy"
SAMPLING = "healpix"

def num_points2nside(num_points):   
    nside = int(np.sqrt(num_points / 12))
    if 12 * nside**2 != num_points:
        nside = int(np.sqrt(num_points / 12)) + 1
    return nside

def s2fft_2d_to_dense(coeffs_2d):
    """
    Converts s2fft's 2D padded format (L, 2L-1) to dense 1D format ((L+1)^2,).
    Removes the zero-padding.
    """
    L = coeffs_2d.shape[0]
    dense_size = L * L
    coeffs_1d = np.zeros(dense_size, dtype=coeffs_2d.dtype)
    
    # s2fft storage: coeffs[l, m_index] where m_index = m + (L-1)
    # dense storage: index = l^2 + l + m
    
    idx = 0
    # s2fft (JAX) usually typically has shape (L, 2L-1)
    # Verify shape aligns with L
    m_offset = coeffs_2d.shape[1] // 2 
    
    for l in range(L):
        # Extract m from -l to +l
        row = coeffs_2d[l]
        # Valid m indices in the row are [m_offset - l : m_offset + l + 1]
        valid_coeffs = row[m_offset - l : m_offset + l + 1]
        
        # Place into dense vector
        coeffs_1d[idx : idx + len(valid_coeffs)] = valid_coeffs
        idx += len(valid_coeffs)
        
    return coeffs_1d

def dense_to_s2fft_2d(coeffs_1d, L):
    """
    Restores dense 1D format to s2fft's 2D padded format (L, 2L-1).
    """
    # Create the 2D grid filled with zeros
    # Standard s2fft shape is often (L, 2L-1)
    shape_2d = (L, 2*L - 1)
    coeffs_2d = np.zeros(shape_2d, dtype=coeffs_1d.dtype)
    m_offset = shape_2d[1] // 2
    
    idx = 0
    for l in range(L):
        num_m = 2 * l + 1
        # Extract from dense
        valid_coeffs = coeffs_1d[idx : idx + num_m]
        
        # Place into 2D grid
        coeffs_2d[l, m_offset - l : m_offset + l + 1] = valid_coeffs
        idx += num_m
        
    return coeffs_2d

def _ensure_coeff_format(coeff, lmax):
    """
    Ensure coefficients are in the correct 2D format for s2fft.inverse.
    s2fft expects coefficients in shape [L+1, 2*L+1].
    """
    if len(coeff.shape) == 1:
        # Reshape 1D coefficients to 2D format
        coeff_2d = np.zeros((lmax + 1, 2 * lmax + 1), dtype=coeff.dtype)
        idx_1d = 0
        for ell in range(lmax + 1):
            for m in range(-ell, ell + 1):
                if idx_1d < len(coeff):
                    coeff_2d[ell, lmax + m] = coeff[idx_1d]
                    idx_1d += 1
        return coeff_2d
    return coeff
    
def fit_ref_points_SH(mesh, origins, N_coarse, lmax, device='cuda'):

    nside = num_points2nside(N_coarse)
    theta_train, phi_train = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32).flatten() # Warp indices are usually flattened
    
    # Create the BVH Mesh object once for efficiency
    wp_points = wp.array(vertices, dtype=wp.vec3, device=device)
    wp_indices = wp.array(triangles, dtype=int, device=device)
    
    wp_mesh = wp.Mesh(
        points=wp_points,
        indices=wp_indices,
        velocities=None
    )
    coeffs_list = []
    
    # Pre-allocate ray arrays to avoid re-allocating memory every loop
    num_samples = len(theta_train)
    distances_wp = wp.zeros(num_samples, dtype=float, device=device)
    
    # Pre-calculate directions (they don't change if theta/phi are constant)
    directions = np.stack([
        np.sin(theta_train)*np.cos(phi_train),
        np.sin(theta_train)*np.sin(phi_train),
        np.cos(theta_train)
    ], axis=1).astype(np.float32)
    directions_wp = wp.array(directions, dtype=wp.vec3, device=device)

    for idx, origin in tqdm.tqdm(enumerate(origins), total=len(origins), desc="Fitting coefficients..."):
        
        # We pass the pre-built wp_mesh to the function
        distances = launch_ray_cast(
            wp_mesh, origin, directions_wp, distances_wp, num_samples
        )
        
        coeffs = s2fft.forward(distances, lmax, nside=nside, sampling=SAMPLING, method=METHOD)
        coeffs_list.append(coeffs)
        
    return coeffs_list

def generate_candidate_points_vec(skeleton_points, coeffs, N_recon, lmax):
   
    nside = num_points2nside(N_recon)
    n_pix = hp.nside2npix(nside)
    theta_recon, phi_recon = hp.pix2ang(nside, np.arange(n_pix))
    
    all_points = []
    all_indices = []
    
    print("Generating candidate points (looping over skeletons)...")
    # TODO: Batch this
    for idx, origin in tqdm.tqdm(enumerate(skeleton_points), total=len(skeleton_points), desc="1. Generating candidates..."):
        
        coeff = _ensure_coeff_format(coeffs[idx], lmax)
        f_recon = s2fft.inverse(coeff, lmax, nside=nside, sampling=SAMPLING, method=METHOD)
        points = spherical_to_cartesian(f_recon, theta_recon, phi_recon, origin)
        indices = np.full(len(points), idx, dtype=int)
        all_points.append(points)
        all_indices.append(indices)
        
    candidate_points = np.concatenate(all_points, axis=0)
    generating_indices = np.concatenate(all_indices, axis=0)
    
    return candidate_points, generating_indices

def smooth_coefficients_list(coeffs_list, lmax, method='gaussian', sigma=None, 
                              cutoff_degree=None, decay_rate=2.0, verbose=True):

    smoothed_coeffs_list = []
    for coeffs in coeffs_list:
        smoothed = _smooth_sh_coefficients(coeffs, lmax)
        smoothed_coeffs_list.append(smoothed)
    
    return smoothed_coeffs_list

def _smooth_sh_coefficients(coeffs, lmax):
    """
    Smooth SH coefficients by applying a low-pass filter.
    """
    is_list_format = isinstance(coeffs, list)
    # Convert JAX arrays to NumPy for mutability
    if is_list_format:
        # List format: [array(2*l+1) for l in range(lmax+1)]
        coeffs_smooth = []
        for c in coeffs:
            # Check if it's a JAX array and convert to numpy
            if hasattr(c, '__array__'):
                coeffs_smooth.append(np.array(c, copy=True))
            else:
                coeffs_smooth.append(c.copy())
    else:
        # 2D array format
        if hasattr(coeffs, '__array__'):
            coeffs_smooth = np.array(coeffs, copy=True)
        else:
            coeffs_smooth = coeffs.copy()

    # sinc(π*l/lmax) / (π*l/lmax)
    for l in range(1, min(lmax + 1, len(coeffs))):
        x = np.pi * l / lmax
        weight = np.sin(x) / x if x != 0 else 1.0
        if is_list_format:
            coeffs_smooth[l] = coeffs_smooth[l] * weight
        else:
            coeffs_smooth[l, :] = coeffs_smooth[l, :] * weight

    return coeffs_smooth

def get_1d_symmetry_mask(L):
    """
    Generates a boolean mask for 1D Dense SH coefficients where m >= 0.
    Input L is the BANDWIDTH (sqrt of array size).
    Array size is expected to be L*L.
    """
    total_coeffs = L * L
    mask = np.zeros(total_coeffs, dtype=bool)
    
    for l in range(L):
        # In Dense 1D format (l^2 + l + m), the block for degree l 
        # starts at index l^2 and has length 2l+1.
        block_start = l**2
        
        # We want m >= 0.
        m_zero_local_idx = l
        start_idx = block_start + m_zero_local_idx
        end_idx = block_start + 2*l + 1
        
        mask[start_idx : end_idx] = True
        
    return mask

def pack_real_coeffs(coeffs, lmax=None):
    """
    Reduces shape from (..., L*L) to Packed 1D.
    Discards m < 0 terms.
    Expects input coeffs to be FLATTENED/DENSE (N, 4096), not (N, 64, 127).
    """
    # 1. Infer L from the last dimension (should be L^2)
    dense_size = coeffs.shape[-1]
    L = int(np.sqrt(dense_size))
    
    # Safety check
    if L * L != dense_size:
        raise ValueError(f"Input array size {dense_size} is not a perfect square. "
                         "pack_real_coeffs expects 1D Dense format (L^2).")
    
    # 2. Get the correct 1D mask
    mask = get_1d_symmetry_mask(L)
    
    # 3. Apply mask to the last dimension
    return coeffs[..., mask]

def unpack_real_coeffs(packed, lmax=None):
    """
    Restores full shape (..., L*L) from packed format.
    Reconstructs m < 0 terms using conjugate symmetry.
    """
    # 1. Infer L from packed size
    # Packed size K = L(L+1)/2. Solve for L: L^2 + L - 2K = 0
    K = packed.shape[-1]
    # Quadratic formula positive root: L = (-1 + sqrt(1 + 8K)) / 2
    L = int((-1 + np.sqrt(1 + 8 * K)) / 2)
    
    total_coeffs = L * L
    
    # Prepare output shape
    out_shape = list(packed.shape)
    out_shape[-1] = total_coeffs
    full_coeffs = np.zeros(out_shape, dtype=packed.dtype)
    
    # 2. Fill m >= 0
    mask = get_1d_symmetry_mask(L)
    full_coeffs[..., mask] = packed
    
    # 3. Restore m < 0
    for l in range(L):
        center = l**2 + l  # Global index of m=0
        for m in range(1, l + 1):
            pos_idx = center + m
            neg_idx = center - m
            
            # Conjugate Symmetry: f_{l,-m} = (-1)^m * conj(f_{l,m})
            sign = -1 if (m % 2 != 0) else 1
            full_coeffs[..., neg_idx] = sign * np.conj(full_coeffs[..., pos_idx])
            
    return full_coeffs