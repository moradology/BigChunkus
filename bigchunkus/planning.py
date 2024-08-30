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
        merged_dataset = xr.concat(self.input_datasets, dim=dim, **concat_args)

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

        return ConcatChunkPlanner(self.input_datasets, merged_dataset, concat_dim_ranges, offsets)


    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        raise ValueError("Multiple datasets without a defined merge strategy! Please use a merge method first.")


class ConcatChunkPlanner(ChunkPlanner):
    def __init__(self, source_datasets: List[xr.Dataset], merged_dataset: xr.Dataset, concat_dim_ranges: List[Tuple[int, int]], offsets: List[int]):
        self.source_datasets = source_datasets
        self.merged_dataset = merged_dataset
        self.concat_dim_ranges = concat_dim_ranges
        self.offsets = offsets

    def generate_zarr_key(self, variable_name: str, chunk_indices: List[int]) -> str:
        return f"{variable_name}/" + ".".join(map(str, chunk_indices))

    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        chunk_map = {}

        for var_name in self.merged_dataset.data_vars:
            var_chunk_map = {}

            chunk_indices_list = []
            for dim, chunk_size in chunk_definition.items():
                chunk_indices = list(range(0, self.merged_dataset.sizes[dim], chunk_size))
                chunk_indices_list.append(chunk_indices)

            for chunk_indices in itertools.product(*chunk_indices_list):
                zarr_key = self.generate_zarr_key(var_name, chunk_indices)
                chunk_slices = {}

                for dim, chunk_index in zip(chunk_definition.keys(), chunk_indices):
                    dim_chunk_slices = []
                    current_chunk_start = chunk_index * chunk_definition[dim]
                    current_chunk_end = min(current_chunk_start + chunk_definition[dim], self.merged_dataset.sizes[dim])

                    # Check if the dimension has been concatenated (spanning across datasets)
                    is_concatenated_dim = any(
                        start < current_chunk_end and end > current_chunk_start
                        for start, end in self.concat_dim_ranges
                    )

                    if is_concatenated_dim:
                        for i, (start, end) in enumerate(self.concat_dim_ranges):
                            if end > current_chunk_start and start < current_chunk_end:
                                slice_start = max(start, current_chunk_start) - start
                                slice_end = min(end, current_chunk_end) - start
                                slice_tuple = (i, slice_start, slice_end)
                                if slice_tuple not in dim_chunk_slices:
                                    dim_chunk_slices.append(slice_tuple)
                    else:
                        dim_chunk_slices.append((0, current_chunk_start, current_chunk_end))

                    chunk_slices[dim] = dim_chunk_slices

                    # Also create a top-level key for the dimension, e.g., "time/0"
                    dim_key = f"{dim}/{chunk_index}"
                    if dim_key not in chunk_map:
                        chunk_map[dim_key] = dim_chunk_slices
                    else:
                        chunk_map[dim_key].extend(dim_chunk_slices)
                        chunk_map[dim_key] = list(set(chunk_map[dim_key]))  # Deduplicate entries

                var_chunk_map[zarr_key] = chunk_slices

            chunk_map[var_name] = var_chunk_map

        return chunk_map
