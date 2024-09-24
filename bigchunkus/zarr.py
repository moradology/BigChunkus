from typing import List, Dict

def build_zarr_key_generator(variable_name: str, chunk_definition: Dict[str, int], variable_dims: List[str]):
    """Build and return a Zarr key generator function with validation performed once."""

    if len(variable_dims) == 0:
        raise ValueError("variable_dims cannot be empty")
        
    if not all(dim in chunk_definition for dim in variable_dims):
        missing_dims = [dim for dim in variable_dims if dim not in chunk_definition]
        raise KeyError(f"Chunk definition missing dimensions: {missing_dims}")

    # Return a closure that generates the Zarr key, skipping validation above
    def generate_zarr_key(chunk_indices: List[int]) -> str:
        """Generate the Zarr key based on validated input (optimized for repeated use)."""
        if len(chunk_indices) != len(variable_dims):
            raise ValueError(f"Mismatched lengths: {len(chunk_indices)} chunk indices, but {len(variable_dims)} variable dimensions.")

        adjusted_indices = [ci // chunk_definition[dim] for ci, dim in zip(chunk_indices, variable_dims)]
        return f"{variable_name}/" + ".".join(map(str, adjusted_indices))

    return generate_zarr_key