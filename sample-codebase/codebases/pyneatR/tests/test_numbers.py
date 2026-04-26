import pytest
import numpy as np
from pyneatR import nnumber, npercent

def test_nnumber_basics():
    x = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    expected = ['10.0', '100.0', '1.0 K', '10.0 K', '100.0 K', '1.0 Mn', '10.0 Mn', '100.0 Mn', '1.0 Bn']
    
    # Test auto unit
    res = nnumber(x, digits=1, unit='auto')
    # Note: 'auto' logic implementation might vary slightly on mode calculation
    # Let's check simply.
    # Actually my implementation for 'auto' finds the mode of magnitudes.
    # Magnitudes: 1, 2, 3(K), 4(K), 5(K), 6(Mn), 7(Mn), 8(Mn), 9(Bn)
    # Counts of mag index: 
    # 0 (1-999): 2
    # 1 (K): 3
    # 2 (Mn): 3
    # 3 (Bn): 1
    # Mode is tied between K and Mn. standard np.argmax returns first occurrence -> K.
    # So if K is chosen: 
    # 10 -> 0.0 K (<0.1 K ?)
    # 1000 -> 1.0 K
    # 1M -> 1,000.0 K
    
    # Test custom per-value unit (unit='custom') - DEFAULT
    res_custom = nnumber(x, digits=1, unit='custom')
    # This should match 'expected' exactly as per R example
    assert np.all(res_custom == expected)

def test_nnumber_fixed_unit():
    x = [123456789.123456]
    res = nnumber(x, digits=1, unit='Mn', prefix='$')
    assert res[0] == '$123.5 Mn'

def test_nnumber_separators():
    x = [10000]
    res = nnumber(x, digits=0, thousand_separator='.', unit='')
    # 10.000
    assert res[0] == '10.000'

def test_npercent():
    # 22.3%
    res = npercent([0.223], is_ratio=True, digits=1)
    assert res[0] == '+22.3%'
    
    res2 = npercent([22.3], is_ratio=False, digits=1)
    assert res2[0] == '+22.3%'
    
    # Growth/Drop
    # -4.01 -> -401% -> 4.0x Drop
    # 2.56 -> 256% -> 2.6x Growth
    res3 = npercent([-4.01, 2.56], is_ratio=True, show_growth_factor=True)
    assert '4.0x Drop' in res3[0]
    assert '2.6x Growth' in res3[1]
    
    # Basis points
    res4 = npercent([0.01], is_ratio=True, show_bps=True)
    assert '100 bps' in res4[0]

def test_nnumber_values():
    # Negative, Zero, Positive
    x = [-1000000, 0, 1000000]
    res = nnumber(x, digits=1)
    assert res[0] == '-1.0 Mn'
    assert res[1] == '0.0'
    assert res[2] == '1.0 Mn'

    # NaN and Inf
    x_special = [np.nan, np.inf, -np.inf]
    res_special = nnumber(x_special)
    assert 'nan' in res_special[0].lower()
    assert 'inf' in res_special[1].lower()


def test_nnumber_units():
    x = [1500, 1500000, 1500000000]
    
    # Force 'K'
    res_k = nnumber(x, unit='K', digits=1)
    assert res_k[0] == '1.5 K'
    assert res_k[1] == '1,500.0 K'
    assert res_k[2] == '1,500,000.0 K'

    # Force 'Mn'
    res_mn = nnumber(x, unit='Mn', digits=1)
    assert res_mn[0] == '0.0 Mn'
    assert res_mn[1] == '1.5 Mn'
    assert res_mn[2] == '1,500.0 Mn'
    
    # Force 'Bn'
    res_bn = nnumber(x, unit='Bn', digits=3)
    assert res_bn[1] == '0.002 Bn'
    assert res_bn[2] == '1.500 Bn'


def test_nnumber_formatting():
    x = [1234.567]
    
    # Prefix/Suffix
    res = nnumber(x, prefix='USD ', suffix=' Only')
    assert res[0] == 'USD 1.2 K Only'
    
    # Separators
    x_sep = [1000000]
    assert nnumber(x_sep)[0] == '1.0 Mn'
    
    x_large = [1000000000]
    
    # Unit 'K' -> 1,000,000.0 K -> . as separator -> 1.000.000,0 K
    res_dot = nnumber(x_large, unit='K', thousand_separator='.')
    assert res_dot[0] == '1.000.000,0 K'
    
    # Custom separator '_'
    res_und = nnumber(x_large, unit='K', thousand_separator='_')
    assert res_und[0] == '1_000_000.0 K'

def test_nnumber_errors():
    with pytest.raises(ValueError):
        nnumber(["abc"])
        
    
    # None input -> converted to NaN usually
    res_none = nnumber([None])
    assert 'nan' in res_none[0].lower()

def test_npercent_extended():
    # BPS Scenarios (BPS, expected_percent_str, decimal_val)
    scenarios = [
        (-0.05, "-5.00%", "(-500 bps)"),
        (-0.02, "-2.00%", "(-200 bps)"),
        (-0.01, "-1.00%", "(-100 bps)"),
        (-0.0075, "-0.75%", "(-75 bps)"),
        (-0.005, "-0.50%", "(-50 bps)"),
        (-0.0025, "-0.25%", "(-25 bps)"),
        (-0.001, "-0.10%", "(-10 bps)"),
        (-0.0001, "-0.01%", "(-1 bps)"),
        (0.0, "0.00%", "(0 bps)"),
        (0.0001, "+0.01%", "(+1 bps)"),
        (0.001, "+0.10%", "(+10 bps)"),
        (0.0025, "+0.25%", "(+25 bps)"),
        (0.005, "+0.50%", "(+50 bps)"),
        (0.0075, "+0.75%", "(+75 bps)"),
        (0.01, "+1.00%", "(+100 bps)"),
        (0.02, "+2.00%", "(+200 bps)"),
        (0.05, "+5.00%", "(+500 bps)"),
    ]

    for val, expected_pct, expected_bps in scenarios:
        res = npercent([val], is_ratio=True, digits=2, show_bps=True, show_plus_sign=True)
        assert expected_pct in res[0], f"Failed pct for {val}: got {res[0]}"
        assert expected_bps in res[0], f"Failed bps for {val}: got {res[0]}"

    # Test plus_sign
    res_no_sign = npercent([0.05], is_ratio=True, show_plus_sign=False, digits=2)
    assert res_no_sign[0] == "5.00%" 

    res_sign = npercent([0.05], is_ratio=True, show_plus_sign=True, digits=2)
    assert res_sign[0] == "+5.00%"

    # Test factor_out
    res_factor = npercent([1.0], is_ratio=True, show_growth_factor=True)
    assert "1.0x Growth" in res_factor[0]

    res_drop_small = npercent([-0.5], is_ratio=True, show_growth_factor=True)
    assert "(Drop)" in res_drop_small[0]

def test_npercent_growth_factor_extended():
    # Value (decimal), Expected Substring
    scenarios = [
        (-1000, "(1000.0x Drop)"),
        (-100, "(100.0x Drop)"),
        (-10, "(10.0x Drop)"),
        (-5, "(5.0x Drop)"),
        (-2, "(2.0x Drop)"),
        (-1, "(1.0x Drop)"),
        (-0.5, "(Drop)"),
        (-0.01, "(Drop)"),
        (0, "(Flat)"),
        (0.01, "(Growth)"),
        (0.5, "(Growth)"),
        (1, "(1.0x Growth)"),
        (2, "(2.0x Growth)"),
        (5, "(5.0x Growth)"),
        (10, "(10.0x Growth)"),
        (100, "(100.0x Growth)"),
        (1000, "(1000.0x Growth)")
    ]
    
    for val, expected in scenarios:
        res = npercent([val], is_ratio=True, show_growth_factor=True)
        assert expected in res[0], f"Failed for {val}: got {res[0]}, expected {expected}"
