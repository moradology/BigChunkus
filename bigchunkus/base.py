from abc import ABC, abstractmethod
import itertools
from typing import Dict, List, Union, Tuple
import xarray as xr

class BaseChunkPlanner(ABC):
    def __init__(self, *datasets: xr.Dataset):
        self.source_datasets = datasets

    @classmethod
    def from_datasets(cls, *datasets: xr.Dataset):
        if not datasets:
            raise ValueError("At least one dataset must be provided.")
        elif len(datasets) > 1:
            from .unmerged import UnmergedChunkPlanner
            return UnmergedChunkPlanner(*datasets)
        else:
            from .single import SingleSourceChunkPlanner
            return SingleSourceChunkPlanner(datasets[0])

    @abstractmethod
    def map_chunks(self, chunk_definition: Dict[str, int]) -> dict:
        pass

