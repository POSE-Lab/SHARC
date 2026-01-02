import os

def create_output_structure(input_mesh_path: str, base_output_dir: str) -> dict:
    """Create organized output folder structure."""
    mesh_name = os.path.splitext(os.path.basename(input_mesh_path))[0]
    root_dir = os.path.join(base_output_dir, mesh_name)
    
    paths = {
        'root': root_dir,
        'input': os.path.join(root_dir, 'input'),
        'ref_points': os.path.join(root_dir, 'ref_points'),
        'coeffs': os.path.join(root_dir, 'coeffs'),
        'reconstruction': os.path.join(root_dir, 'reconstruction'),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    print(f"Created output structure at: {root_dir}")
    return paths