from .single import SingleSourceChunkPlanner

from abc import ABC, abstractmethod
import itertools
from typing import Dict, List, Union, Tuple
import xarray as xr

class ChunkPlanner(ABC):
    def __init__(self, *datasets: xr.Dataset):
        self.input_datasets = datasets

    @classmethod
    def from_datasets(cls, *datasets: xr.Dataset):
        if not datasets:
            raise ValueError("At least one dataset must be provided.")
        elif len(datasets) > 1:
            return UnmergedChunkPlanner(*datasets)
        else:
            return SingleSourceChunkPlanner(datasets[0])

    @abstractmethod
    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        pass

class UnmergedChunkPlanner(ChunkPlanner):
    def concat(self, dim: str, **concat_args) -> 'ConcatChunkPlanner':
        # Concatenate the datasets
        #merged_dataset = xr.concat(self.input_datasets, dim=dim, **concat_args)
        merged_dataset = xr.concat(self.input_datasets, dim=dim, coords='minimal', compat='override')

        # Calculate ranges and offsets for the concatenated dimension
        concat_dim_ranges = []
        offsets = []
        current_offset = 0

        for dataset in self.input_datasets:
            dim_size = dataset.sizes[dim]
            start = current_offset
            end = start + dim_size
            concat_dim_ranges.append((start, end))
            offsets.append(current_offset)
            current_offset = end

        # Pass the concatenated dimension name to ConcatChunkPlanner
        return ConcatChunkPlanner(self.input_datasets, merged_dataset, concat_dim_ranges, offsets, concat_dim_name=dim)
    

    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        raise ValueError("Multiple datasets without a defined merge strategy! Please use a merge method first.")

class ConcatChunkPlanner(ChunkPlanner):
    def __init__(self, source_datasets: List[xr.Dataset], merged_dataset: xr.Dataset, concat_dim_ranges: List[Tuple[int, int]], offsets: List[int], concat_dim_name: str):
        self.source_datasets = source_datasets
        self.merged_dataset = merged_dataset
        self.concat_dim_ranges = concat_dim_ranges
        self.offsets = offsets
        self.concat_dim_name = concat_dim_name  # This is the dimension we're concatenating along

    def generate_zarr_key(self, variable_name: str, chunk_indices: List[int], chunk_definition: Dict[str, int]) -> str:
        """Generate the Zarr key for a variable and its chunk indices"""
        print(f"variable_name: {variable_name}, chunk_indices: {chunk_indices}")

        # Adjust the indices for all dimensions, based on chunk size
        adjusted_indices = [ci // chunk_definition[dim] for ci, dim in zip(chunk_indices, self.merged_dataset.dims)]
        
        print(f"adjusted_indices: {adjusted_indices}\n")
        
        # Return the Zarr key with adjusted indices for all dimensions
        return f"{variable_name}/" + ".".join(map(str, adjusted_indices))


    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        """Map output chunks to input slices without duplication"""
        chunk_map = {}

        for var_name in self.merged_dataset.data_vars:
            # Calculate chunk indices for each dimension
            chunk_indices_list = []
            for dim, chunk_size in chunk_definition.items():
                chunk_indices = list(range(0, self.merged_dataset.sizes[dim], chunk_size))
                chunk_indices_list.append(chunk_indices)

            # Iterate through combinations of chunk indices
            for chunk_indices in itertools.product(*chunk_indices_list):
                zarr_key = self.generate_zarr_key(var_name, chunk_indices, chunk_definition)
                chunk_slices = {}

                # Iterate over each dimension and compute chunk slices
                for dim, chunk_index in zip(chunk_definition.keys(), chunk_indices):
                    dim_chunk_slices = []
                    current_chunk_start = chunk_index
                    current_chunk_end = min(current_chunk_start + chunk_definition[dim], self.merged_dataset.sizes[dim])

                    # Handle concatenated dimension
                    if dim == self.concat_dim_name:
                        print(f"\nHandling concatenated dimension '{dim}'")
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

    # def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
    #     """Map output chunks to input slices without duplication"""
    #     chunk_map = {}

    #     for var_name in self.merged_dataset.data_vars:
    #         # Calculate chunk indices for each dimension
    #         chunk_indices_list = []
    #         for dim, chunk_size in chunk_definition.items():
    #             chunk_indices = list(range(0, self.merged_dataset.sizes[dim], chunk_size))
    #             chunk_indices_list.append(chunk_indices)

    #         # Iterate through combinations of chunk indices
    #         for chunk_indices in itertools.product(*chunk_indices_list):
    #             zarr_key = self.generate_zarr_key(var_name, chunk_indices, chunk_definition)
    #             chunk_slices = {}

    #             # Process each dimension only once per chunk
    #             processed_dims = set()

    #             # Iterate over each dimension and compute chunk slices
    #             for dim, chunk_index in zip(chunk_definition.keys(), chunk_indices):
    #                 if dim in processed_dims:
    #                     continue  # Skip already processed dimensions

    #                 dim_chunk_slices = []
    #                 current_chunk_start = chunk_index
    #                 current_chunk_end = min(current_chunk_start + chunk_definition[dim], self.merged_dataset.sizes[dim])

    #                 # Handle concatenated dimension
    #                 if dim == self.concat_dim_name:
    #                     print(f"\nHandling concatenated dimension '{dim}'")
    #                     for i, (start, end) in enumerate(self.concat_dim_ranges):
    #                         if end > current_chunk_start and start < current_chunk_end:
    #                             slice_start = max(start, current_chunk_start) - start
    #                             slice_end = min(end, current_chunk_end) - start
    #                             slice_tuple = (i, slice_start, slice_end)
    #                             dim_chunk_slices.append(slice_tuple)

    #                     chunk_key_index = current_chunk_start // chunk_definition[dim]
    #                     dim_key = f"{dim}/{chunk_key_index}"
    #                 else:
    #                     # Handle non-concatenated dimensions normally
    #                     slice_tuple = (0, current_chunk_start, current_chunk_end)
    #                     dim_chunk_slices.append(slice_tuple)
    #                     dim_key = f"{dim}/{chunk_index}"

    #                 # Add slices to the chunk map (avoid adding the same dimension slices multiple times)
    #                 if dim_key not in chunk_map:
    #                     chunk_map[dim_key] = dim_chunk_slices
    #                 else:
    #                     # Only add new slices that haven't been processed yet
    #                     chunk_map[dim_key].extend(
    #                         [s for s in dim_chunk_slices if s not in chunk_map[dim_key]]
    #                     )

    #                 chunk_slices[dim] = dim_chunk_slices
    #                 processed_dims.add(dim)  # Mark the dimension as processed

    #             # Assign Zarr key with correct chunk slices
    #             chunk_map[zarr_key] = chunk_slices

    #     return chunk_map
