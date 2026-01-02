
import numpy as np
import os
from scipy.spatial.distance import pdist

def calc_max_dist(points):
    """Compute the diameter (maximum pairwise distance) of a point cloud."""
    # Compute all pairwise distances and return the maximum
    return pdist(points).max()

def move_origin_to_lower_left(mesh_vertices):
    """
    Translates mesh vertices so that the bounding box's minimum corner is at the origin.
    This function computes the bounding box of the input vertices and translates
    all vertices such that the minimum corner (lower-left in 2D, or minimum x,y,z
    in 3D) of the bounding box is positioned at the coordinate system origin (0,0,0).
    Args:
        mesh_vertices (numpy.ndarray): Array of vertex coordinates with shape (N, D)
                                    where N is the number of vertices and D is the
                                    dimensionality (2D or 3D).
    Returns:
        numpy.ndarray: Translated vertices with the same shape as input, where
                    the bounding box minimum corner is at the origin.
    Example:
        >>> vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> translated = move_origin_to_lower_left(vertices)
        >>> # The minimum corner [1, 2, 3] becomes [0, 0, 0]
    """
   
    # Compute the dimensions of the bounding box
    bbox_dimensions = np.amax(mesh_vertices, axis=0) - np.amin(mesh_vertices, axis=0)
    # Compute the minimum corner of the bounding box
    min_corner = np.amin(mesh_vertices, axis=0)
    # Translate vertices to move the origin to the lower left corner of the bounding box
    translated_vertices = mesh_vertices - min_corner
    return translated_vertices

def fitModel2UnitSphere(points, buffer=1.0, scale=1.0):
    """Fits the model to a unit sphere using centroid-based scaling.
    Points can be also padded using a buffer.

    Args:
        points (NDArray): Point cloud points.
        buffer (float): Buffer factor for sphere (>1 shrinks the final sphere).
        scale (float): Final scale factor.

    Returns:
        tuple: (normalized_points, scale, radius, center) - Normalized point cloud points,
               scale factor, original radius, and sphere center.
    """
    # Compute centroid
    center = np.mean(points, axis=0)
    
    # Recenter mesh at the centroid
    points_centered = points - center
    
    # Compute max distance from centroid (radius)
    radius = np.max(np.linalg.norm(points_centered, axis=1))
    
    # Apply buffer (divide to make buffer>1 shrink the sphere relative to unit size)
    effective_radius = radius / buffer
    
    # Scale to unit sphere
    points_normalized = points_centered / effective_radius
    
    # Apply final scale
    points_normalized *= scale
    
    return points_normalized, scale, effective_radius, center

def spherical_to_cartesian(r, theta, phi, origin=np.zeros(3)):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z).
    
    Parameters
    ----------
    r : array_like
        Radial distance
    theta : array_like
        Polar angle (0 to π)
    phi : array_like
        Azimuthal angle (0 to 2π)
    origin : array_like, shape (3,), optional
        Origin offset, default is (0,0,0)
    
    Returns
    -------
    points : np.ndarray, shape (..., 3)
        Cartesian coordinates
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    origin = np.asarray(origin)
    
    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    
    points = np.stack([x, y, z], axis=-1)
    return points + origin

def cartesian_to_spherical(points, origin=np.zeros(3)):
    """
    Convert Cartesian coordinates (x, y, z) to spherical (r, theta, phi).
    
    Parameters
    ----------
    points : array_like, shape (..., 3)
        Cartesian coordinates
    origin : array_like, shape (3,), optional
        Origin offset, default is (0,0,0)
    
    Returns
    -------
    r : np.ndarray
        Radial distance
    theta : np.ndarray
        Polar angle (0 to π)
    phi : np.ndarray
        Azimuthal angle (0 to 2π)
    """
    points = np.asarray(points)
    origin = np.asarray(origin)
    
    # Shift points relative to origin
    rel_points = points - origin
    
    x = rel_points[..., 0]
    y = rel_points[..., 1]
    z = rel_points[..., 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-12), -1, 1))
    phi = np.arctan2(y, x) % (2 * np.pi)
    
    return r, theta, phi