import pytest
import numpy as np
from pyneatR import set_locale, get_locale, reset_locale, Locale, LOCALES
from pyneatR.locale import resolve_locale, format_grouped_number


class TestLocaleRegistry:
    def test_predefined_locales_exist(self):
        assert "en_IN" in LOCALES
        assert "en_US" in LOCALES
        assert "de_DE" in LOCALES

    def test_locale_is_frozen(self):
        loc = LOCALES["en_IN"]
        with pytest.raises(Exception):
            loc.name = "something_else"

    def test_en_IN_properties(self):
        loc = LOCALES["en_IN"]
        assert loc.thousand_separator == ","
        assert loc.decimal_separator == "."
        assert loc.grouping == (3, 2)
        assert loc.default_currency == "INR"
        assert loc.timezone_label == "IST"
        assert "lakh" in loc.unit_labels
        assert "crore" in loc.unit_labels

    def test_en_US_properties(self):
        loc = LOCALES["en_US"]
        assert loc.grouping == (3,)
        assert loc.default_currency == "USD"
        assert loc.timezone_label == "EST"

    def test_de_DE_properties(self):
        loc = LOCALES["de_DE"]
        assert loc.thousand_separator == "."
        assert loc.decimal_separator == ","
        assert loc.currency_position == "suffix"
        assert loc.percent_space is True


class TestGlobalLocale:
    def setup_method(self):
        reset_locale()

    def teardown_method(self):
        reset_locale()

    def test_default_is_none(self):
        assert get_locale() is None

    def test_set_and_get(self):
        set_locale("en_IN")
        loc = get_locale()
        assert loc is not None
        assert loc.name == "en_IN"

    def test_reset(self):
        set_locale("en_US")
        assert get_locale() is not None
        reset_locale()
        assert get_locale() is None

    def test_set_invalid_locale(self):
        with pytest.raises(ValueError, match="Unknown locale"):
            set_locale("xx_XX")


class TestResolveLocale:
    def setup_method(self):
        reset_locale()

    def teardown_method(self):
        reset_locale()

    def test_none_returns_global(self):
        assert resolve_locale(None) is None
        set_locale("en_IN")
        assert resolve_locale(None).name == "en_IN"

    def test_string_lookup(self):
        loc = resolve_locale("en_US")
        assert loc.name == "en_US"

    def test_locale_object_passthrough(self):
        loc = LOCALES["de_DE"]
        assert resolve_locale(loc) is loc

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            resolve_locale("xx_XX")

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            resolve_locale(42)


class TestIndianGrouping:
    """Test the Indian number grouping algorithm."""

    def test_small_number(self):
        assert format_grouped_number("100", (3, 2), ",") == "100"

    def test_thousands(self):
        assert format_grouped_number("1000", (3, 2), ",") == "1,000"

    def test_ten_thousands(self):
        assert format_grouped_number("10000", (3, 2), ",") == "10,000"

    def test_lakh(self):
        assert format_grouped_number("100000", (3, 2), ",") == "1,00,000"

    def test_ten_lakhs(self):
        assert format_grouped_number("1000000", (3, 2), ",") == "10,00,000"

    def test_crore(self):
        assert format_grouped_number("10000000", (3, 2), ",") == "1,00,00,000"

    def test_ten_crores(self):
        assert format_grouped_number("100000000", (3, 2), ",") == "10,00,00,000"

    def test_arab(self):
        assert format_grouped_number("1000000000", (3, 2), ",") == "1,00,00,00,000"

    def test_standard_grouping(self):
        assert format_grouped_number("1000000", (3,), ",") == "1,000,000"

    def test_single_digit(self):
        assert format_grouped_number("5", (3, 2), ",") == "5"

    def test_dot_separator(self):
        assert format_grouped_number("10000000", (3,), ".") == "10.000.000"


class TestNnumberWithLocale:
    """Test nnumber with locale-aware formatting."""

    def setup_method(self):
        reset_locale()

    def teardown_method(self):
        reset_locale()

    def test_indian_grouping_raw_number(self):
        from pyneatR import nnumber
        result = nnumber(1_00_00_000, unit='', locale='en_IN', digits=0)
        assert result == "1,00,00,000"

    def test_indian_unit_labels(self):
        from pyneatR import nnumber
        result = nnumber(1_50_00_000, locale='en_IN')
        assert "Cr" in result

    def test_indian_lakh(self):
        from pyneatR import nnumber
        result = nnumber(1_50_000, locale='en_IN')
        assert "L" in result

    def test_us_locale_default(self):
        from pyneatR import nnumber
        result = nnumber(1500000, locale='en_US')
        assert "M" in result

    def test_eu_separators(self):
        from pyneatR import nnumber
        result = nnumber(1234.56, unit='', locale='de_DE', digits=2)
        # Should use . as thousand sep, , as decimal sep
        assert "," in result  # decimal separator

    def test_backward_compatible_no_locale(self):
        """Without locale, existing behavior should be unchanged."""
        from pyneatR import nnumber
        x = [10, 100, 1000, 10000, 100000, 1000000]
        expected = ['10.0', '100.0', '1.0 K', '10.0 K', '100.0 K', '1.0 Mn']
        result = nnumber(x, digits=1, unit='custom')
        assert np.all(result == expected)

    def test_global_locale_fallback(self):
        from pyneatR import nnumber
        set_locale("en_IN")
        result = nnumber(1_00_00_000, unit='', digits=0)
        assert "1,00,00,000" in result


class TestNpercentBugFixes:
    """Verify the npercent bugs are fixed."""

    def test_no_duplicate_bps(self):
        from pyneatR import npercent
        result = npercent([0.01], is_ratio=True, show_bps=True, digits=2)
        # Count occurrences of "bps)" — should be exactly 1
        count = result[0].count("bps)")
        assert count == 1, f"Expected 1 occurrence of 'bps)' but got {count}: {result[0]}"

    def test_no_duplicate_formatting(self):
        """Just ensure npercent doesn't crash and produces expected output."""
        from pyneatR import npercent
        result = npercent([0.5], is_ratio=True, digits=1)
        assert result[0] == "+50.0%"
