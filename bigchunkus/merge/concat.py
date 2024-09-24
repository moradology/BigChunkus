from ..base import BaseChunkPlanner
from ..zarr import build_zarr_key_generator

import itertools
from typing import Dict, List, Union, Tuple
import xarray as xr

class ConcatChunkPlanner(BaseChunkPlanner):
    def __init__(self, source_datasets: List[xr.Dataset], merged_dataset: xr.Dataset, concat_dim_ranges: List[Tuple[int, int]], offsets: List[int], concat_dim_name: str):
        self.source_datasets = source_datasets
        self.merged_dataset = merged_dataset
        self.concat_dim_ranges = concat_dim_ranges
        self.offsets = offsets
        self.concat_dim_name = concat_dim_name  # This is the dimension we're concatenating along

    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        """Map output chunks to input slices without duplication"""
        chunk_map = {}

        # Merge the user-defined chunk definition with the dataset's chunk sizes
        dataset_dim_sizes = dict(self.merged_dataset.sizes)
        dataset_chunk_sizes = dict(self.merged_dataset.chunks)
        chunk_definition = {**dataset_dim_sizes, **dataset_chunk_sizes, **chunk_definition}

        print("Final chunk definition with fallbacks:", chunk_definition)

        for var_name, var_data in self.merged_dataset.data_vars.items():
            # Get the specific dimensions for this variable
            variable_dims = var_data.dims
            
            # Calculate chunk indices for the dimensions relevant to this variable
            chunk_indices_list = []
            for dim in variable_dims:
                chunk_size = chunk_definition[dim]
                chunk_indices = list(range(0, self.merged_dataset.sizes[dim], chunk_size))
                chunk_indices_list.append(chunk_indices)

            # Validate arguments at this point and produce closure to avoid revalidation
            generate_zarr_key = build_zarr_key_generator(
                var_name,
                chunk_definition,
                variable_dims
            )
            # Iterate through combinations of chunk indices
            for chunk_indices in itertools.product(*chunk_indices_list):
                zarr_key = generate_zarr_key(chunk_indices)
                chunk_slices = {}

                # Iterate over each dimension relevant to this variable and compute chunk slices
                for dim, chunk_index in zip(variable_dims, chunk_indices):
                    dim_chunk_slices = []
                    current_chunk_start = chunk_index
                    current_chunk_end = min(current_chunk_start + chunk_definition[dim], self.merged_dataset.sizes[dim])

                    # Handle concatenation dimension
                    if dim == self.concat_dim_name:
                        for i, (start, end) in enumerate(self.concat_dim_ranges):
                            if end > current_chunk_start and start < current_chunk_end:
                                slice_start = max(start, current_chunk_start) - start
                                slice_end = min(end, current_chunk_end) - start
                                slice_tuple = (i, slice_start, slice_end)
                                dim_chunk_slices.append(slice_tuple)

                        chunk_key_index = current_chunk_start // chunk_definition[dim]
                        dim_key = f"{dim}/{chunk_key_index}"
                    else:
                        # Handle non-concatenated dimensions normally
                        slice_tuple = (0, current_chunk_start, current_chunk_end)
                        dim_chunk_slices.append(slice_tuple)
                        dim_key = f"{dim}/{chunk_index}"

                    # Add slices to the chunk map (avoid adding the same dimension slices multiple times)
                    if dim_key not in chunk_map:
                        chunk_map[dim_key] = dim_chunk_slices
                    else:
                        # Only add new slices that haven't been processed yet
                        chunk_map[dim_key].extend(
                            [s for s in dim_chunk_slices if s not in chunk_map[dim_key]]
                        )

                    chunk_slices[dim] = dim_chunk_slices

                # Assign Zarr key with correct chunk slices
                chunk_map[zarr_key] = chunk_slices

        return chunk_map
