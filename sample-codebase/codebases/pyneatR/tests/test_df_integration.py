import pytest
import numpy as np
import pandas as pd
import polars as pl
from pyneatR import nnumber, ndate, nstring, nday, npercent, ntimestamp

def test_pandas_series():
    # Numbers
    s = pd.Series([1000, 2000, 3000])
    res = nnumber(s)
    # Should return numpy array of strings
    assert isinstance(res, np.ndarray) or isinstance(res, list) or isinstance(res, pd.Series)
    assert res[0] == "1.0 K"
    
    # Dates
    d = pd.to_datetime(["2023-01-01", "2023-01-02"])
    res_d = ndate(d, show_weekday=False)
    assert res_d[0] == "Jan 01, 2023"

def test_pandas_dataframe_column():
    df = pd.DataFrame({"vals": [1000000, 2000000]})
    res = nnumber(df["vals"], unit='Mn')
    assert res[0] == "1.0 Mn"

def test_polars_series():
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")
        
    s = pl.Series("vals", [1000, 2000, 3000])
    # nnumber handles numpy-like. Polars series might need conversion or nnumber handles it via np.asanyarray
    res = nnumber(s)
    
    # If nnumber returns numpy array, that's fine.
    # We check content.
    assert "1.0 K" in res[0] 

def test_polars_edge_cases():
    try:
        import polars as pl
    except ImportError:
        pytest.skip("Polars not installed")

    # Polars with nulls
    s = pl.Series("vals", [1000, None, 3000])
    res = nnumber(s)
    assert len(res) == 3

def test_comprehensive_dataframe():
    data = {
        "user_review": [
            "Great app!!! Used 5 times in 2 days, cost $12.99 :)", "Terrible!! Crashed 999 times @#$%",
            "Average service, 3/10, paid $45.67", "Loved it <3 used 100 times, saved $1000!!!",
            "Worst experience ever!!! -1 stars", "", None, "Okay-ish… used once, paid $0.99",
            "Numbers 1234567890 !!! ??? ###", "Good but expensive, $99999.99!!!",
            "Refunded 50%, not happy :(", "Used on 2024-01-01 @ 12:30:45, works fine",
            "🔥🔥🔥 10/10 would recommend $$$", "Error code 500, retry count 7",
            "Cheap!!! only $0.01 unbelievable", "Overcharged by $1000000!!!",
            None, "Mixed feelings... paid $12.00 twice",
            "Special chars only !!!@@@###$$$", "Final test review 42 times"
        ],
        "mcc_code": [
            5411, 5812, 5732, 5999, 4111, 1234, None, 7999, 4899, 9999,
            5311, 5814, 5651, 0, 1, 8888, None, 3000, 7000, 5555
        ],
        "revenue": [
            12.99, 0.0, 45.67, 1000.00, -5.00, np.nan, None, 0.99, 123456.78, 99999.99,
            50.00, 10.00, 5.55, 0.01, 0.01, 1000000.00, np.nan, 24.00, 0.00, 42.42
        ],
        "conversion_rate": [
            0.22, 0.00, 0.45, 0.95, -0.10, np.nan, None, 0.01, 0.88, 1.50,
            0.50, 0.30, 0.99, 0.001, 0.02, 2.00, np.nan, 0.24, 0.00, 0.42
        ],
        "date": [
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            None, "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10",
            "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15",
            "2024-01-16", None, "2024-01-18", "2024-01-19", "2024-01-20"
        ],
        "timestamp": [
            "2024-01-01 10:15:30", "2024-01-02 11:00:00", "2024-01-03 09:45:10",
            "2024-01-04 23:59:59", "2024-01-05 00:00:01", None, "2024-01-07 14:22:33",
            "2024-01-08 08:08:08", "2024-01-09 12:12:12", "2024-01-10 16:45:00",
            "2024-01-11 10:10:10", "2024-01-12 12:30:45", "2024-01-13 01:01:01",
            "2024-01-14 18:18:18", "2024-01-15 20:20:20", "2024-01-16 22:22:22",
            None, "2024-01-18 06:06:06", "2024-01-19 09:09:09", "2024-01-20 23:23:23"
        ]
    }
    
    # === PANDAS ===
    df_pd = pd.DataFrame(data)
    df_pd["date"] = pd.to_datetime(df_pd["date"], errors="coerce")
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"], errors="coerce")
    
    # Test valid execution and basics
    # nnumber
    res_rev = nnumber(df_pd["revenue"], unit='custom')
    assert "1.0 Mn" in res_rev[15]
    
    # npercent
    res_kp = npercent(df_pd["conversion_rate"], is_ratio=True)
    assert "+22.0%" in res_kp[0]
    
    # ndate
    res_dt = ndate(df_pd["date"])
    assert "Jan 01, 2024" in res_dt[0]
    assert "NaT" in str(res_dt[5])
    
    # ntimestamp
    res_ts = ntimestamp(df_pd["timestamp"])
    assert "10H 15M 30S AM" in res_ts[0]
    
    # nstring
    res_str = nstring(df_pd["user_review"], remove_specials=True)
    assert "Great app" in res_str[0]
    assert "!!!" not in res_str[0]
    
    # === POLARS ===
    try:
        import polars as pl
        df_pl = pl.DataFrame(data)
        # Note: Polars handles dates differently, often needs casting if passed as strings
        # But pyneatR takes array-like and converts to numpy.
        # Polars Series -> Numpy array might need care for dates/timestamps if they are Objects or proper types.
        # If they are strings in Polars, they stay strings in Numpy. ndate expects datetime64[D] or strings it can cast.
        # Let's see if plain strings work.
        
        # Numbers
        res_pl_rev = nnumber(df_pl["revenue"])
        assert "1.0 Mn" in res_pl_rev[15]
        
        # Percent
        res_pl_cv = npercent(df_pl["conversion_rate"])
        assert "+22.0%" in res_pl_cv[0]
        
        # Dates (Passed as strings from data dict -> Polars Utf8 -> Numpy Object/String -> ndate conversion)
        res_pl_dt = ndate(df_pl["date"])
        assert "Jan 01, 2024" in res_pl_dt[0]
        
        # Timestamps
        res_pl_ts = ntimestamp(df_pl["timestamp"])
        assert "10H 15M 30S AM" in res_pl_ts[0]
        
    except ImportError:
        pass # Skip if polars not available

