from bigchunkus.planning import ChunkPlanner, UnmergedChunkPlanner, ConcatChunkPlanner

import numpy as np
import xarray as xr

def test_concat_chunk_planner_with_zarr_keys():
    # Create two small datasets
    ds1 = xr.Dataset({
        "var": (("time", "x"), np.array([[1, 2, 3], [4, 5, 6]]))
    }, coords={"time": [0, 1], "x": [10, 20, 30]})

    ds2 = xr.Dataset({
        "var": (("time", "x"), np.array([[7, 8, 9], [10, 11, 12]]))
    }, coords={"time": [2, 3], "x": [10, 20, 30]})

    # Initialize the UnmergedChunkPlanner with the two datasets
    unmerged_planner = UnmergedChunkPlanner(ds1, ds2)

    # Concatenate along the 'time' dimension
    concat_planner = unmerged_planner.concat(dim='time')

    # Define the chunking strategy for the output Zarr store
    chunk_definition = {
        "time": 1,  # Each chunk in 'time' dimension will contain 1 value
        "x": 3  # Each chunk in 'x' dimension will contain 3 values (entire 'x' dimension)
    }

    # Map the output chunks to input slices
    chunk_map = concat_planner.map_chunks(chunk_definition)

    # Print out the chunk map to understand what has been generated
    print("Chunk Map:", chunk_map)

    # Expected Zarr keys for the chunks
    expected_chunk_map = {
        "var/0.0": {
            "time": [(0, 0, 1)],  # First chunk in time dimension from ds1
            "x": [(0, 0, 3)]      # Full x dimension for that chunk
        },
        "var/1.0": {
            "time": [(0, 1, 2)],  # Second chunk in time dimension from ds1
            "x": [(0, 0, 3)]      # Full x dimension for that chunk
        },
        "var/2.0": {
            "time": [(1, 0, 1)],  # First chunk in time dimension from ds2
            "x": [(0, 0, 3)]      # Full x dimension for that chunk
        },
        "var/3.0": {
            "time": [(1, 1, 2)],  # Second chunk in time dimension from ds2
            "x": [(0, 0, 3)]      # Full x dimension for that chunk
        }
    }

    # Assertions to verify correct behavior
    assert "var/0.0" in chunk_map
    assert "var/1.0" in chunk_map
    assert "var/2.0" in chunk_map
    assert "var/3.0" in chunk_map

    # Verify the Zarr keys and their corresponding slices for both 'time' and 'x' dimensions
    assert chunk_map["var/0.0"] == expected_chunk_map["var/0.0"]
    assert chunk_map["var/1.0"] == expected_chunk_map["var/1.0"]
    assert chunk_map["var/2.0"] == expected_chunk_map["var/2.0"]
    assert chunk_map["var/3.0"] == expected_chunk_map["var/3.0"]


def test_concat_chunk_planner_chunk_spanning_multiple_source_files():
    # Create two small datasetsj12j12
    ds1 = xr.Dataset({
        "var": (("time", "x"), np.array([[1, 2, 3], [4, 5, 6]]))
    }, coords={"time": [0, 1], "x": [10, 20, 30]})

    ds2 = xr.Dataset({
        "var": (("time", "x"), np.array([[7, 8, 9], [10, 11, 12]]))
    }, coords={"time": [2, 3], "x": [10, 20, 30]})

    # Initialize the UnmergedChunkPlanner with the two datasets
    unmerged_planner = UnmergedChunkPlanner(ds1, ds2)

    # Concatenate along the 'time' dimension
    concat_planner = unmerged_planner.concat(dim='time')

    # Define the chunking strategy for the output Zarr store
    chunk_definition = {
        "time": 3,  # Each chunk in 'time' dimension will contain 3 values (spanning both ds1 and ds2)
        "x": 3  # Each chunk in 'x' dimension will contain 3 values (entire 'x' dimension)
    }

    # Map the output chunks to input slices
    chunk_map = concat_planner.map_chunks(chunk_definition)

    # Print out the chunk map to understand what has been generated
    print("Chunk Map:", chunk_map)

    # Expected Zarr keys for the chunks
    expected_chunk_map = {
        "time/0": [(0, 0, 2), (1, 0, 1)],  # Time chunk spanning ds1 and ds2
        "time/1": [(1, 1, 2)],  # Remaining time chunk from ds2
        "x/0": [(0, 0, 3)],  # Chunk for x, same across both datasets
        "var/0.0": {  # The first Zarr key for the variable
            "time": [(0, 0, 2), (1, 0, 1)],  # Time chunks spanning ds1 and ds2
            "x": [(0, 0, 3)]  # Full x dimension from both datasets
        },
        "var/1.0": {  # The second Zarr key for the variable
            "time": [(1, 1, 2)],  # Fully within ds2
            "x": [(0, 0, 3)]  # Same x dimension as above
        }
    }

    # Assertions to verify correct behavior
    assert "time/0" in chunk_map
    assert "time/1" in chunk_map
    assert "x/0" in chunk_map
    assert "var/0.0" in chunk_map
    assert "var/1.0" in chunk_map

    # Verify the time and x dimensions
    assert chunk_map["time/0"] == expected_chunk_map["time/0"]
    assert chunk_map["time/1"] == expected_chunk_map["time/1"]
    assert chunk_map["x/0"] == expected_chunk_map["x/0"]
    assert chunk_map["var/0.0"] == expected_chunk_map["var/0.0"]
    assert chunk_map["var/1.0"] == expected_chunk_map["var/1.0"]

def test_concat_chunk_planner_with_varied_dataset_sizes():
    # Create four datasets with different sizes
    ds1 = xr.Dataset({
        "var": (("time", "x"), np.array([[1, 2, 3], [4, 5, 6]]))
    }, coords={"time": [0, 1], "x": [10, 20, 30]})

    ds2 = xr.Dataset({
        "var": (("time", "x"), np.array([[7, 8, 9]]))
    }, coords={"time": [2], "x": [10, 20, 30]})

    ds3 = xr.Dataset({
        "var": (("time", "x"), np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]))
    }, coords={"time": [3, 4, 5], "x": [10, 20, 30]})

    ds4 = xr.Dataset({
        "var": (("time", "x"), np.array([[19, 20, 21]]))
    }, coords={"time": [6], "x": [10, 20, 30]})

    unmerged_planner = UnmergedChunkPlanner(ds1, ds2, ds3, ds4)
    concat_planner = unmerged_planner.concat(dim='time')
    chunk_definition = {
        "time": 3,
        "x": 3
    }
    chunk_map = concat_planner.map_chunks(chunk_definition)

    expected_chunk_map = {
        "time/0": [(0, 0, 2), (1, 0, 1)],  # chunk spanning ds1 and ds2
        "time/1": [(2, 0, 3)],  # chunk from ds3
        "time/2": [(3, 0, 1)],  # chunk from ds4
        "x/0": [(0, 0, 3)],
        "var/0.0": {
            "time": [(0, 0, 2), (1, 0, 1)],
            "x": [(0, 0, 3)]
        },
        "var/1.0": {
            "time": [(2, 0, 3)],
            "x": [(0, 0, 3)]
        },
        "var/2.0": {
            "time": [(3, 0, 1)],
            "x": [(0, 0, 3)]
        }
    }

    assert "time/0" in chunk_map
    assert "time/1" in chunk_map
    assert "time/2" in chunk_map
    assert "x/0" in chunk_map
    assert "var/0.0" in chunk_map
    assert "var/1.0" in chunk_map
    assert "var/2.0" in chunk_map

    # Verify the time and x dimensions
    assert chunk_map["time/0"] == expected_chunk_map["time/0"]
    assert chunk_map["time/1"] == expected_chunk_map["time/1"]
    assert chunk_map["time/2"] == expected_chunk_map["time/2"]
    assert chunk_map["x/0"] == expected_chunk_map["x/0"]
    assert chunk_map["var/0.0"] == expected_chunk_map["var/0.0"]
    assert chunk_map["var/1.0"] == expected_chunk_map["var/1.0"]
    assert chunk_map["var/2.0"] == expected_chunk_map["var/2.0"]
