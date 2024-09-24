import pytest

from bigchunkus.zarr import build_zarr_key_generator

def test_generate_zarr_key_basic():
    # Basic case with 2 dimensions
    variable_name = "temperature"
    chunk_indices = [25, 50]
    chunk_definition = {"time": 10, "lat": 25}
    variable_dims = ["time", "lat"]

    expected_key = "temperature/2.2"
    generate_zarr_key = build_zarr_key_generator(variable_name, chunk_definition, variable_dims)
    assert generate_zarr_key(chunk_indices) == expected_key, \
        "Basic case failed to generate correct Zarr key."

def test_generate_zarr_key_multi_dimensional():
    # Multi-dimensional case with 3 dimensions
    variable_name = "precipitation"
    chunk_indices = [30, 100, 45]
    chunk_definition = {"time": 10, "lat": 50, "lon": 50}
    variable_dims = ["time", "lat", "lon"]

    expected_key = "precipitation/3.2.0"
    generate_zarr_key = build_zarr_key_generator(variable_name, chunk_definition, variable_dims)
    assert generate_zarr_key(chunk_indices) == expected_key, \
        "Multi-dimensional case failed to generate correct Zarr key."

def test_generate_zarr_key_single_dimension():
    # Single-dimensional case
    variable_name = "pressure"
    chunk_indices = [100]
    chunk_definition = {"time": 25}
    variable_dims = ["time"]

    expected_key = "pressure/4"
    generate_zarr_key = build_zarr_key_generator(variable_name, chunk_definition, variable_dims)
    assert generate_zarr_key(chunk_indices) == expected_key, \
        "Single-dimensional case failed to generate correct Zarr key."

def test_generate_zarr_key_exact_multiple():
    # Case where chunk index is an exact multiple of chunk size
    variable_name = "temperature"
    chunk_indices = [20, 100]
    chunk_definition = {"time": 10, "lat": 50}
    variable_dims = ["time", "lat"]

    expected_key = "temperature/2.2"  # 20 // 10 = 2, 100 // 50 = 2
    generate_zarr_key = build_zarr_key_generator(variable_name, chunk_definition, variable_dims)
    assert generate_zarr_key(chunk_indices) == expected_key, \
        "Exact multiple case failed to generate correct Zarr key."

def test_generate_zarr_key_mismatched_dimensions():
    # Mismatched chunk_indices and variable_dims lengths
    variable_name = "wind"
    chunk_indices = [10, 50]
    chunk_definition = {"time": 10, "lat": 25, "lon": 50}
    variable_dims = ["time", "lat", "lon"]

    generate_zarr_key = build_zarr_key_generator(variable_name, chunk_definition, variable_dims)
    with pytest.raises(ValueError):
        generate_zarr_key(chunk_indices)

def test_build_zarr_generator_missing_dimension_in_chunk_definition():
    # Missing dimension in the chunk definition
    variable_name = "humidity"
    chunk_indices = [30, 45]
    chunk_definition = {"time": 10}  # Missing "lat" dimension
    variable_dims = ["time", "lat"]

    with pytest.raises(KeyError):
        build_zarr_key_generator(variable_name, chunk_definition, variable_dims)
