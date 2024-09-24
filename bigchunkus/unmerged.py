from .merge import ConcatChunkPlanner
from .base import BaseChunkPlanner

from typing import Dict
import xarray as xr

class UnmergedChunkPlanner(BaseChunkPlanner):

    def _order_datasets_by_dim(self, dim: str):
        """
        Reorder datasets by the first value in the 'time' coordinate for each dataset.
        """
        # Sort the datasets based on the starting time value in the 'time' coordinate.
        self.source_datasets = sorted(
            self.source_datasets, 
            key=lambda ds: ds.indexes[dim][0]
        )

    def concat(self, dim: str, manually_ordered: bool = False, **concat_args) -> ConcatChunkPlanner:
        if not manually_ordered:
            for ds in self.source_datasets:
                if dim not in ds.indexes:
                    raise ValueError(
                        f"Expected dimension '{dim}' not in dataset indexes.\n"
                        "If indexes aren't loaded and the order of concatenation is known "
                        "ahead of time, you may want to set 'manually_ordered' to True and "
                        "ensure that datasets are supplied to the UnmergedChunkPlanner in the "
                        "anticipated order."
                    )
            self._order_datasets_by_dim(dim)

        merged_dataset = xr.concat(
            self.source_datasets,
            dim=dim,
            coords='minimal',
            compat='override',
            data_vars='minimal',
            **concat_args
        )

        # Calculate ranges and offsets for the concatenated dimension
        concat_dim_ranges = []
        offsets = []
        current_offset = 0

        for dataset in self.source_datasets:
            dim_size = dataset.sizes[dim]
            start = current_offset
            end = start + dim_size
            concat_dim_ranges.append((start, end))
            offsets.append(current_offset)
            current_offset = end

        # Pass the concatenated dimension name to ConcatChunkPlanner
        return ConcatChunkPlanner(self.source_datasets, merged_dataset, concat_dim_ranges, offsets, concat_dim_name=dim)
    

    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        raise ValueError("Multiple datasets without a defined merge strategy! Please use a merge method first.")
