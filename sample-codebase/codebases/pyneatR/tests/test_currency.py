import pytest
import numpy as np
from pyneatR import ncurrency, get_currency_symbol, CURRENCY_SYMBOLS
from pyneatR.locale import reset_locale


class TestCurrencyRegistry:
    """Test the currency symbol registry."""

    def test_major_currencies_present(self):
        major = ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "CHF"]
        for code in major:
            assert code in CURRENCY_SYMBOLS, f"Missing currency: {code}"

    def test_inr_symbol(self):
        assert CURRENCY_SYMBOLS["INR"] == "₹"

    def test_usd_symbol(self):
        assert CURRENCY_SYMBOLS["USD"] == "$"

    def test_eur_symbol(self):
        assert CURRENCY_SYMBOLS["EUR"] == "€"

    def test_gbp_symbol(self):
        assert CURRENCY_SYMBOLS["GBP"] == "£"

    def test_jpy_symbol(self):
        assert CURRENCY_SYMBOLS["JPY"] == "¥"

    def test_crypto_symbols(self):
        assert CURRENCY_SYMBOLS["BTC"] == "₿"
        assert CURRENCY_SYMBOLS["ETH"] == "Ξ"

    def test_south_asian_currencies(self):
        assert "PKR" in CURRENCY_SYMBOLS
        assert "LKR" in CURRENCY_SYMBOLS
        assert "BDT" in CURRENCY_SYMBOLS
        assert "NPR" in CURRENCY_SYMBOLS

    def test_registry_has_50_plus_currencies(self):
        assert len(CURRENCY_SYMBOLS) >= 50


class TestGetCurrencySymbol:
    def test_valid_code(self):
        assert get_currency_symbol("USD") == "$"
        assert get_currency_symbol("INR") == "₹"

    def test_case_insensitive(self):
        assert get_currency_symbol("usd") == "$"
        assert get_currency_symbol("inr") == "₹"

    def test_invalid_code(self):
        with pytest.raises(ValueError, match="Unknown currency code"):
            get_currency_symbol("XYZ")


class TestNcurrency:
    def setup_method(self):
        reset_locale()

    def teardown_method(self):
        reset_locale()

    def test_usd_basic(self):
        result = ncurrency(1500, currency="USD", unit='')
        assert "$" in result
        assert "1,500" in result

    def test_inr_basic(self):
        result = ncurrency(1500, currency="INR", unit='', locale="en_IN")
        assert "₹" in result

    def test_inr_crore(self):
        result = ncurrency(1_50_00_000, currency="INR", locale="en_IN")
        assert "₹" in result
        assert "Cr" in result

    def test_inr_lakh(self):
        result = ncurrency(1_50_000, currency="INR", locale="en_IN")
        assert "₹" in result
        assert "L" in result

    def test_eur_suffix(self):
        result = ncurrency(1500, currency="EUR", unit='', locale="de_DE")
        assert "€" in result
        # EUR should be suffix for de_DE
        assert result.strip().endswith("€")

    def test_eur_decimal_comma(self):
        result = ncurrency(1234.56, currency="EUR", unit='', locale="de_DE", digits=2)
        # Should use comma as decimal separator
        assert "," in result

    def test_scalar_returns_string(self):
        result = ncurrency(1000, currency="USD")
        assert isinstance(result, str)

    def test_array_returns_array(self):
        result = ncurrency([1000, 2000], currency="USD")
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_negative_value(self):
        result = ncurrency(-1000, currency="USD", unit='')
        assert "-" in result

    def test_nan_handling(self):
        result = ncurrency(float('nan'), currency="USD")
        assert "NaN" in result

    def test_inf_handling(self):
        result = ncurrency(float('inf'), currency="USD")
        assert "∞" in result

    def test_zero(self):
        result = ncurrency(0, currency="USD", unit='')
        assert "$" in result
        assert "0" in result

    def test_custom_digits(self):
        result = ncurrency(1234.567, currency="USD", unit='', digits=3)
        assert "567" in result

    def test_default_currency_from_locale(self):
        result = ncurrency(1000, locale="en_IN", unit='')
        assert "₹" in result

    def test_indian_grouping_in_currency(self):
        result = ncurrency(1_00_00_000, currency="INR", unit='', locale="en_IN", digits=0)
        assert "₹" in result
        assert "1,00,00,000" in result
