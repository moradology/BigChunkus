from typing import Dict, List, Mapping, Tuple

def merge_chunk_definitions(dict1: Dict[str, int], dict2: Mapping[str, int]) -> Dict[str, int]:
    dict2 = dict(dict2)
    return {key: dict1.get(key, dict2.get(key)) for key in dict1.keys() | dict2.keys()}

def get_zarr_key(var_name: str, chunk_indices: Tuple[int, ...], chunk_sizes: Dict[str, int], var_dims: List[str]) -> str:
    key_parts = [var_name]
    for dim, index in zip(var_dims, chunk_indices):
        if dim in chunk_sizes:
            key_parts.append(str(index // chunk_sizes[dim]))
        else:
            key_parts.append('0')  # For dimensions not explicitly chunked (like 'bnds')
    return '/'.join(key_parts)