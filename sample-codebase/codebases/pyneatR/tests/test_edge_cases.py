import pytest
import numpy as np
from pyneatR import nnumber, npercent, nstring, ndate

def test_empty_input():
    # Test all functions with empty input
    assert len(nnumber([])) == 0
    assert len(npercent([])) == 0
    assert len(nstring([])) == 0
    assert len(ndate([])) == 0

def test_nan_handling_numbers():
    x = [10.0, np.nan, 1000.0]
    res = nnumber(x, unit='auto')
    # Expect nan to be handled gracefully, likely as 'nan' or generic formatting
    # The current implementation of numpy string conversion might need checking
    # But usually nnumber returns generic string for nan or fail?
    # Let's assume it returns 'nan' or similar string representation.
    assert len(res) == 3
    # We check if it doesn't crash at least

def test_inf_handling_numbers():
    x = [np.inf, -np.inf, 1000.0]
    res = nnumber(x)
    assert len(res) == 3
    
def test_nan_handling_dates():
    # NaT handling
    d = np.array(['2023-01-01', 'NaT'], dtype='datetime64[D]')
    res = ndate(d)
    assert len(res) == 2
    # Check if NaT is handled (usually "NaT" string or generic)

def test_none_input():
    # Lists with None
    x = [10, None, 100]
    # This might fail if the function doesn't convert to float/array safely
    # If it fails, we will know we need to fix the code.
    try:
        res = nnumber(x)
        assert len(res) == 3
    except Exception as e:
        pytest.fail(f"nnumber failed with None input: {e}")

def test_mixed_types_strings():
    x = ["hello", 123, None]
    res = nstring(x)
    assert len(res) == 3
    assert res[0] == "hello"
    # 123 might be converted to "123"
    # None handling depends on implementation
