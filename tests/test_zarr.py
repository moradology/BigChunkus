from bigchunkus.zarr import extract_zarr_key_info, generate_zarr_keys, get_data_for_zarr_key

import numpy as np
import xarray as xr

# ===================
# Zarr key generation
# ===================

def test_single_variable_single_dimension_generation():
    data = xr.Dataset({"var": (("x",), [1, 2, 3, 4, 5, 6])}, coords={"x": [0, 1, 2, 3, 4, 5]})
    chunked_data = data.chunk({"x": 2})
    zarr_keys = generate_zarr_keys(chunked_data)
    assert zarr_keys == {
        "var": ["var/0", "var/2", "var/4"]
    }

def test_multiple_variables_single_dimension_generation():
    data = xr.Dataset({
        "var1": (("x",), [1, 2, 3, 4]),
        "var2": (("x",), [5, 6, 7, 8])
    }, coords={"x": [0, 1, 2, 3]})
    chunked_data = data.chunk({"x": 2})
    zarr_keys = generate_zarr_keys(chunked_data)
    assert zarr_keys == {
        "var1": ["var1/0", "var1/2"],
        "var2": ["var2/0", "var2/2"]
    }

def test_single_variable_multiple_dimensions_generation():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2], [3, 4], [5, 6], [7, 8]])
    }, coords={"x": [0, 1, 2, 3], "y": [0, 1]})
    chunked_data = data.chunk({"x": 2, "y": 1})
    zarr_keys = generate_zarr_keys(chunked_data)
    assert zarr_keys == {
        "var": ["var/0.0", "var/0.1", "var/2.0", "var/2.1"]
    }

def test_multiple_variables_multiple_dimensions_generation():
    data = xr.Dataset({
        "var1": (("x", "y"), [[1, 2], [3, 4], [5, 6], [7, 8]]),
        "var2": (("x", "y"), [[8, 7], [6, 5], [4, 3], [2, 1]])
    }, coords={"x": [0, 1, 2, 3], "y": [0, 1]})
    chunked_data = data.chunk({"x": 2, "y": 1})
    zarr_keys = generate_zarr_keys(chunked_data)
    assert zarr_keys == {
        "var1": ["var1/0.0", "var1/0.1", "var1/2.0", "var1/2.1"],
        "var2": ["var2/0.0", "var2/0.1", "var2/2.0", "var2/2.1"]
    }

def test_no_chunking_generation():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2], [3, 4], [5, 6], [7, 8]])
    }, coords={"x": [0, 1, 2, 3], "y": [0, 1]})
    # No chunking applied
    zarr_keys = generate_zarr_keys(data)
    assert zarr_keys == {
        "var": ["var/0.0"]
    }

def test_single_element_dimension_generation():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2]])
    }, coords={"x": [0], "y": [0, 1]})
    chunked_data = data.chunk({"x": 1, "y": 1})
    zarr_keys = generate_zarr_keys(chunked_data)
    assert zarr_keys == {
        "var": ["var/0.0", "var/0.1"]
    }

def test_non_uniform_chunk_sizes_generation():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    # Use a tuple for non-uniform chunking
    chunked_data = data.chunk({"x": (1, 2), "y": 2})
    zarr_keys = generate_zarr_keys(chunked_data)
    assert zarr_keys == {
        "var": ["var/0.0", "var/0.2", "var/1.0", "var/1.2"]
    }

def test_empty_dataset_generation():
    data = xr.Dataset()
    zarr_keys = generate_zarr_keys(data)
    assert zarr_keys == {}

# ========================
# Zarr key data extraction
# ========================

def test_basic_case_with_uniform_chunking_extraction():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    chunked_data = data.chunk({"x": 2, "y": 2})
    zarr_key = "var/0.0"
    
    chunk_indices, chunk_sizes, dimension_sizes = extract_zarr_key_info(zarr_key, chunked_data)
    
    assert chunk_indices == [0, 0]
    assert chunk_sizes == [2, 2]
    assert dimension_sizes == [3, 3]

def test_non_chunked_dataset_extraction():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    zarr_key = "var/0.0"
    
    chunk_indices, chunk_sizes, dimension_sizes = extract_zarr_key_info(zarr_key, data)
    
    assert chunk_indices == [0, 0]
    assert chunk_sizes == [3, 3]  # Entire dimension is treated as one chunk
    assert dimension_sizes == [3, 3]

def test_non_uniform_chunking_extraction():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    chunked_data = data.chunk({"x": (1, 2), "y": 2})
    zarr_key = "var/0.0"
    
    chunk_indices, chunk_sizes, dimension_sizes = extract_zarr_key_info(zarr_key, chunked_data)
    
    assert chunk_indices == [0, 0]
    assert chunk_sizes == [1, 2]  # First chunk for x is 1, for y it's 2
    assert dimension_sizes == [3, 3]

def test_single_dimension_extraction():
    data = xr.Dataset({
        "var": ("x", [1, 2, 3, 4, 5])
    }, coords={"x": [0, 1, 2, 3, 4]})
    chunked_data = data.chunk({"x": 2})
    zarr_key = "var/0"
    
    chunk_indices, chunk_sizes, dimension_sizes = extract_zarr_key_info(zarr_key, chunked_data)
    
    assert chunk_indices == [0]
    assert chunk_sizes == [2]
    assert dimension_sizes == [5]

def test_small_final_chunk_extraction():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2], [3, 4], [5, 6]])
    }, coords={"x": [0, 1, 2], "y": [0, 1]})
    chunked_data = data.chunk({"x": 2, "y": 2})
    zarr_key = "var/1.0"  # Final chunk for x
    
    chunk_indices, chunk_sizes, dimension_sizes = extract_zarr_key_info(zarr_key, chunked_data)
    
    assert chunk_indices == [1, 0]
    assert chunk_sizes == [1, 2]  # The final chunk in x is smaller (1 instead of 2)
    assert dimension_sizes == [3, 2]

def test_empty_dimension_extraction():
    data = xr.Dataset({
        "var": (("x", "y"), np.empty((0, 0)))
    }, coords={"x": [], "y": []})
    
    zarr_key = "var/0.0"
    
    chunk_indices, chunk_sizes, dimension_sizes = extract_zarr_key_info(zarr_key, data)
    
    assert chunk_indices == [0, 0]
    assert chunk_sizes == [0, 0]
    assert dimension_sizes == [0, 0]

# =======================
# Get data using zarr key
# =======================

def test_basic_chunk_retrieval():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    chunked_data = data.chunk({"x": 2, "y": 2})
    zarr_key = "var/0.0"
    
    data_chunk = get_data_for_zarr_key(chunked_data, zarr_key)
    
    expected = xr.DataArray([[1, 2], [4, 5]], dims=("x", "y"), coords={"x": [0, 1], "y": [0, 1]})
    xr.testing.assert_equal(data_chunk, expected)

def test_non_uniform_chunk_retrieval():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    chunked_data = data.chunk({"x": (1, 2), "y": 2})
    zarr_key = "var/1.0"
    
    data_chunk = get_data_for_zarr_key(chunked_data, zarr_key)
    
    expected = xr.DataArray([[4, 5], [7, 8]], dims=("x", "y"), coords={"x": [1, 2], "y": [0, 1]})
    xr.testing.assert_equal(data_chunk, expected)

def test_final_chunk_retrieval():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2], [3, 4], [5, 6]])
    }, coords={"x": [0, 1, 2], "y": [0, 1]})
    chunked_data = data.chunk({"x": 2, "y": 2})
    zarr_key = "var/1.0"
    
    data_chunk = get_data_for_zarr_key(chunked_data, zarr_key)
    
    expected = xr.DataArray([[5, 6]], dims=("x", "y"), coords={"x": [2], "y": [0, 1]})
    xr.testing.assert_equal(data_chunk, expected)

def test_empty_dimension_retrieval():
    data = xr.Dataset({
        "var": (("x", "y"), np.empty((0, 0)))
    }, coords={"x": [], "y": []})
    zarr_key = "var/0.0"
    
    data_chunk = get_data_for_zarr_key(data, zarr_key)
    
    expected = xr.DataArray(np.empty((0, 0)), dims=("x", "y"), coords={"x": [], "y": []})
    xr.testing.assert_equal(data_chunk, expected)

def test_single_dimension_retrieval():
    data = xr.Dataset({
        "var": ("x", [1, 2, 3, 4, 5])
    }, coords={"x": [0, 1, 2, 3, 4]})
    chunked_data = data.chunk({"x": 2})
    zarr_key = "var/1"
    
    data_chunk = get_data_for_zarr_key(chunked_data, zarr_key)
    
    expected = xr.DataArray([3, 4], dims=("x"), coords={"x": [2, 3]})
    xr.testing.assert_equal(data_chunk, expected)

def test_non_chunked_dataset_retrieval():
    data = xr.Dataset({
        "var": (("x", "y"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    zarr_key = "var/0.0"
    
    data_chunk = get_data_for_zarr_key(data, zarr_key)
    
    expected = xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dims=("x", "y"), coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    xr.testing.assert_equal(data_chunk, expected)
