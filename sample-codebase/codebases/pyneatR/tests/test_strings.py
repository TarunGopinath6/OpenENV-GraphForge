import pytest
import numpy as np
from pyneatR import nstring

def test_nstring_case():
    x = ["hello world", "GOODBYE"]
    res = nstring(x, case='title')
    assert res[0] == "Hello World"
    assert res[1] == "Goodbye"
    
def test_nstring_specials():
    x = ["hello!!! world@"]
    res = nstring(x, remove_specials=True)
    # keep only alnum + space
    assert res[0] == "hello world"

def test_nstring_start_case():
    x = ["all Models are Wrong"]
    res = nstring(x, case='start')
    # All Models Are Wrong
    assert res[0] == "All Models Are Wrong"

def test_nstring_cases():
    x = ["tEst CasE"]
    
    # Lower
    assert nstring(x, case='lower')[0] == "test case"
    # Upper
    assert nstring(x, case='upper')[0] == "TEST CASE"
    # Initcap (Capitalize first char of string)
    assert nstring(x, case='initcap')[0] == "Test case" 
    # Title (Title Case)
    assert nstring(x, case='title')[0] == "Test Case"
    
def test_nstring_cleaning():
    s = "Hello @World! 123"
    
    # Remove specials
    # alphanumeric + space kept. @ and ! removed.
    res = nstring([s], remove_specials=True)
    assert res[0] == "Hello World 123"
    
    # Whitelist
    res_wl = nstring([s], remove_specials=True, keep_chars="!")
    # @ removed, ! kept
    assert res_wl[0] == "Hello World! 123"
    
    # En only (remove non-ascii)
    s_uni = "Hello unicode \u00E9" # é
    res_en = nstring([s_uni], ascii_only=True)
    assert res_en[0] == "Hello unicode" # é removed or replaced? Regex [^\x20-\x7E] -> removed.

def test_nstring_scalars():
    s = " scalar Input "
    res = nstring(s, case='upper')
    # Should strip whitespace too?
    # Logic: space_pattern.sub(" ", s).strip() is applied
    assert res == "SCALAR INPUT"

