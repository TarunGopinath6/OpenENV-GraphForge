import pytest
import numpy as np
import datetime
from pyneatR import f

def test_f_inference_date():
    # date inference
    d = datetime.date(2026, 1, 1)
    assert f(d) == "Jan 01, 2026"
    
    # numpy date inference
    d_np = np.datetime64('2026-01-01', 'D')
    assert f(d_np) == "Jan 01, 2026"

def test_f_inference_ts():
    # timestamp inference — no locale set, no timezone label injected
    ts = datetime.datetime(2025, 11, 9, 12, 7, 48)
    res = f(ts)
    assert "Nov 09, 2025" in res
    assert "12H 07M 48S PM" in res
    assert "(Sun)" in res

def test_f_inference_ts_with_locale():
    # With en_IN locale, IST should be injected
    ts = datetime.datetime(2025, 11, 9, 12, 7, 48)
    res = f(ts, locale="en_IN")
    assert "12H 07M 48S PM IST" in res
    assert "(Sun)" in res


def test_f_inference_number():
    # number inference
    # Note: 1.3 Tn is mathematically correct for 1.345e12 if scaling to Tn
    assert f(1345000000000) == "1.3 Tn"
    assert f(1234.56, digits=2, unit='') == "1,234.56"

def test_f_inference_string():
    # string inference
    s = "all Models are Wrong!!!"
    assert f(s) == "All Models Are Wrong"

def test_f_day():
    d = datetime.date(2026, 1, 1) # Thursday
    assert f(d, format_type='day') == "Thu"

def test_f_percent():
    # +900% (9x growth, 90K basis points)
    # Actually 9.0 ratio -> 900%
    # 9x growth -> check
    # 90K basis points -> 9 * 100 * 100 = 90,000
    res = f(9.0, format_type='percent')
    assert "+900.0%" in res
    assert "9x growth" in res
    assert "90K basis points" in res

def test_f_kwargs():
    # Verify kwargs are passed
    assert f(1234.56, format_type='number', digits=2, thousand_separator='_', unit='') == "1_234.56"
    assert f("hello world", case='upper') == "HELLO WORLD"

def test_f_vectorized():
    x = [1000, 2000]
    res = f(x)
    assert res[0] == "1.0 K"
    assert res[1] == "2.0 K"
    
    d = [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)]
    # Convert to numpy array to ensure consistent inference in test
    res_d = f(np.asanyarray(d, dtype='datetime64[D]'))
    assert res_d[0] == "Jan 01, 2026"
    assert res_d[1] == "Jan 02, 2026"

def test_f_inference_mixed():
    # Mixed numeric/None should infer 'number'
    x = [1000, None, 2000]
    res = f(x)
    assert res[0] == "1.0 K"
    assert res[1] == "nan"
    assert res[2] == "2.0 K"

def test_f_invalid_type():
    with pytest.raises(ValueError):
        f(10, format_type='invalid')
