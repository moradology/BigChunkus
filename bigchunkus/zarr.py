from itertools import product
from typing import List, Dict, Tuple
import xarray as xr


def generate_zarr_keys(dataset: xr.Dataset) -> Dict[str, List[str]]:
    """
    Generate Zarr keys for an xarray.Dataset.

    Parameters:
    - dataset (xr.Dataset): The xarray dataset for which to generate Zarr keys.

    Returns:
    - Dict[str, List[str]]: A dictionary where the keys are variable names and the values
                            are lists of corresponding Zarr keys (filenames).
    """
    zarr_keys = {}

    for var_name, variable in dataset.data_vars.items():
        # Get the chunk sizes for the variable
        chunks = variable.chunks
        
        # Initialize the list to hold the Zarr keys for this variable
        keys = []

        # Generate the indices for each dimension
        def generate_indices(dim_sizes, chunk_sizes):
            all_ranges = []
            for dim_size, chunk_size in zip(dim_sizes, chunk_sizes or [None] * len(dim_sizes)):
                if chunk_size is None:
                    # No chunking, so treat the entire dimension as one chunk
                    all_ranges.append([0])
                elif isinstance(chunk_size, int):
                    all_ranges.append(range(0, dim_size, chunk_size))
                else:
                    # Handle the case where chunk_size is a tuple of sizes
                    indices = []
                    start = 0
                    for size in chunk_size:
                        indices.append(start)
                        start += size
                    all_ranges.append(indices)
            return all_ranges
        
        # Generate all combinations of chunk indices using Cartesian product
        all_combinations = product(*generate_indices(variable.sizes.values(), chunks))
        
        # Iterate over all combinations of chunk indices
        for index_combination in all_combinations:
            # Create a key by joining the indices with dots
            chunk_index_str = ".".join(map(str, index_combination))
            # Combine the variable name with the chunk index string to form the Zarr key
            key = f"{var_name}/{chunk_index_str}"
            keys.append(key)
        
        # Store the generated keys for this variable
        zarr_keys[var_name] = keys
    
    return zarr_keys

def extract_zarr_key_info(zarr_key: str, dataset: xr.Dataset) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract chunk indices, chunk sizes, and dimension sizes from an xarray.Dataset based on a Zarr key.

    Parameters:
    - zarr_key (str): The Zarr key, e.g., 'variable_name/0.0.0'.
    - dataset (xr.Dataset): The xarray.Dataset from which to extract the information.

    Returns:
    - Tuple containing:
        - List of chunk indices corresponding to the Zarr key.
        - List of chunk sizes for each dimension.
        - List of full dimension sizes for the variable.
    """
    # Step 1: Parse the Zarr key
    key_parts = zarr_key.split('/')
    variable_name = key_parts[0]
    chunk_indices = list(map(int, key_parts[1].split('.')))

    # Step 2: Retrieve the DataArray associated with the variable
    data_array = dataset[variable_name]

    # Step 3: Extract chunk sizes and dimension sizes
    if data_array.chunks is not None:
        chunk_sizes = []
        for dim_index, chunks in enumerate(data_array.chunks):
            # Handle the final chunk, which may be smaller
            chunk_index = chunk_indices[dim_index]
            if chunk_index < len(chunks) - 1:
                chunk_sizes.append(chunks[chunk_index])
            else:
                chunk_sizes.append(chunks[-1])  # Final chunk size
    else:
        # If chunks are not defined, treat the entire dimension as a single chunk
        chunk_sizes = list(data_array.shape)

    dimension_sizes = list(data_array.sizes.values())

    return chunk_indices, chunk_sizes, dimension_sizes

def get_data_for_zarr_key(dataset: xr.Dataset, zarr_key: str) -> xr.DataArray:
    """
    Retrieve the data corresponding to a given Zarr key from an xarray.Dataset.

    Parameters:
    - dataset (xr.Dataset): The xarray.Dataset from which to extract the data.
    - zarr_key (str): The Zarr key, e.g., 'variable_name/0.0.0'.

    Returns:
    - xr.DataArray: The subset of the dataset corresponding to the Zarr key.
    """
    chunk_indices, chunk_sizes, dimension_sizes = extract_zarr_key_info(zarr_key, dataset)

    slices = []
    variable_name = zarr_key.split('/')[0]
    data_array = dataset[variable_name]

    for i, (index, size, dim_size) in enumerate(zip(chunk_indices, chunk_sizes, dimension_sizes)):
        if data_array.chunks is None:
            # Handle the case where the dataset is not chunked; return full range
            start, end = 0, dim_size
        else:
            chunks = data_array.chunks[i]  # Get chunks for the ith dimension
            start = sum(chunks[:index])  # Sum up the sizes of all preceding chunks to get the start
            end = min(start + size, dim_size)  # Calculate the end, ensuring it doesn't exceed dimension size
        slices.append(slice(start, end))
    
    return data_array[tuple(slices)]
