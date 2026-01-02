import trimesh
import open3d as o3d

def trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Convert a Trimesh Trimesh object to an Open3D TriangleMesh.

    Parameters:
        tri_mesh (trimesh.Trimesh): The Trimesh mesh to convert.

    Returns:
        o3d.geometry.TriangleMesh: The converted Open3D mesh object.
    """
    # Create an Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    
    # Set vertices and faces
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    
    
    return o3d_mesh