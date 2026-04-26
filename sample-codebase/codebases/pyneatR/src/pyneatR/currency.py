"""
Unicode currency symbol registry and currency formatting for pyneatR.

Provides a hardcoded mapping of ISO 4217 currency codes to their Unicode
symbols, plus the `ncurrency` function for locale-aware currency formatting.
"""
import numpy as np
from typing import Union, Optional, Dict
from .utils import _check_singleton
from .locale import resolve_locale, Locale, format_grouped_number

# ---------------------------------------------------------------------------
# Currency symbol registry — 60+ currencies, zero external dependencies
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS: Dict[str, str] = {
    # ── Major global currencies ──
    "USD": "$",          # U+0024  United States Dollar
    "EUR": "€",          # U+20AC  Euro
    "GBP": "£",          # U+00A3  British Pound
    "JPY": "¥",          # U+00A5  Japanese Yen
    "CNY": "¥",          # U+00A5  Chinese Yuan
    "CHF": "CHF",        #         Swiss Franc (no unique symbol)

    # ── South Asia ──
    "INR": "₹",          # U+20B9  Indian Rupee
    "PKR": "₨",          # U+20A8  Pakistani Rupee
    "LKR": "₨",          # U+20A8  Sri Lankan Rupee
    "NPR": "₨",          # U+20A8  Nepalese Rupee
    "BDT": "৳",          # U+09F3  Bangladeshi Taka
    "MVR": "Rf",         #         Maldivian Rufiyaa

    # ── East & Southeast Asia ──
    "KRW": "₩",          # U+20A9  South Korean Won
    "THB": "฿",          # U+0E3F  Thai Baht
    "VND": "₫",          # U+20AB  Vietnamese Dong
    "PHP": "₱",          # U+20B1  Philippine Peso
    "IDR": "Rp",         #         Indonesian Rupiah
    "MYR": "RM",         #         Malaysian Ringgit
    "SGD": "S$",         #         Singapore Dollar
    "HKD": "HK$",        #         Hong Kong Dollar
    "TWD": "NT$",        #         New Taiwan Dollar
    "MNT": "₮",          # U+20AE  Mongolian Tugrik
    "KHR": "៛",          # U+17DB  Cambodian Riel
    "LAK": "₭",          # U+20AD  Lao Kip
    "MMK": "K",          #         Myanmar Kyat

    # ── Oceania ──
    "AUD": "A$",         #         Australian Dollar
    "NZD": "NZ$",        #         New Zealand Dollar
    "FJD": "FJ$",        #         Fijian Dollar

    # ── Americas ──
    "CAD": "C$",         #         Canadian Dollar
    "MXN": "$",          # U+0024  Mexican Peso
    "BRL": "R$",         #         Brazilian Real
    "ARS": "$",          # U+0024  Argentine Peso
    "CLP": "$",          # U+0024  Chilean Peso
    "COP": "$",          # U+0024  Colombian Peso
    "PEN": "S/.",        #         Peruvian Sol
    "UYU": "$U",         #         Uruguayan Peso
    "CRC": "₡",          # U+20A1  Costa Rican Colón
    "GTQ": "Q",          #         Guatemalan Quetzal
    "JMD": "J$",         #         Jamaican Dollar
    "TTD": "TT$",        #         Trinidad Dollar

    # ── Europe (non-Euro) ──
    "SEK": "kr",         #         Swedish Krona
    "NOK": "kr",         #         Norwegian Krone
    "DKK": "kr",         #         Danish Krone
    "ISK": "kr",         #         Icelandic Króna
    "CZK": "Kč",         #         Czech Koruna
    "HUF": "Ft",         #         Hungarian Forint
    "PLN": "zł",         #         Polish Zloty
    "RON": "lei",        #         Romanian Leu
    "BGN": "лв",         #         Bulgarian Lev
    "HRK": "kn",         #         Croatian Kuna
    "RSD": "din.",       #         Serbian Dinar
    "UAH": "₴",          # U+20B4  Ukrainian Hryvnia
    "GEL": "₾",          # U+20BE  Georgian Lari
    "AMD": "֏",          # U+058F  Armenian Dram
    "TRY": "₺",          # U+20BA  Turkish Lira
    "RUB": "₽",          # U+20BD  Russian Ruble

    # ── Central Asia ──
    "KZT": "₸",          # U+20B8  Kazakhstani Tenge
    "AZN": "₼",          # U+20BC  Azerbaijani Manat
    "UZS": "сўм",        #         Uzbekistani Som
    "TMT": "T",          #         Turkmenistani Manat

    # ── Middle East ──
    "AED": "د.إ",        #         UAE Dirham
    "SAR": "﷼",          # U+FDFC  Saudi Riyal
    "QAR": "﷼",          # U+FDFC  Qatari Riyal
    "KWD": "د.ك",        #         Kuwaiti Dinar
    "BHD": "BD",         #         Bahraini Dinar
    "OMR": "﷼",          # U+FDFC  Omani Rial
    "JOD": "د.ا",        #         Jordanian Dinar
    "ILS": "₪",          # U+20AA  Israeli Shekel
    "IQD": "ع.د",        #         Iraqi Dinar
    "IRR": "﷼",          # U+FDFC  Iranian Rial
    "LBP": "ل.ل",       #         Lebanese Pound
    "SYP": "£S",         #         Syrian Pound

    # ── Africa ──
    "ZAR": "R",          #         South African Rand
    "NGN": "₦",          # U+20A6  Nigerian Naira
    "GHS": "₵",          # U+20B5  Ghanaian Cedi
    "KES": "KSh",        #         Kenyan Shilling
    "TZS": "TSh",        #         Tanzanian Shilling
    "UGX": "USh",        #         Ugandan Shilling
    "EGP": "E£",         #         Egyptian Pound
    "MAD": "MAD",        #         Moroccan Dirham
    "TND": "DT",         #         Tunisian Dinar
    "XOF": "CFA",        #         West African CFA Franc
    "XAF": "FCFA",       #         Central African CFA Franc
    "ETB": "Br",         #         Ethiopian Birr
    "RWF": "FRw",        #         Rwandan Franc
    "MUR": "₨",          # U+20A8  Mauritian Rupee

    # ── Crypto ──
    "BTC": "₿",          # U+20BF  Bitcoin
    "ETH": "Ξ",          # U+039E  Ethereum
}


def get_currency_symbol(currency_code: str) -> str:
    """
    Get the Unicode symbol for an ISO 4217 currency code.
    
    Parameters
    ----------
    currency_code : str
        ISO 4217 currency code (e.g. "USD", "INR", "EUR").
    
    Returns
    -------
    str
        The currency symbol (e.g. "$", "₹", "€").
    
    Raises
    ------
    ValueError
        If the currency code is not in the registry.
    """
    code = currency_code.upper()
    if code not in CURRENCY_SYMBOLS:
        raise ValueError(
            f"Unknown currency code '{code}'. "
            f"Use one of: {', '.join(sorted(CURRENCY_SYMBOLS.keys())[:10])}... "
            f"({len(CURRENCY_SYMBOLS)} total)"
        )
    return CURRENCY_SYMBOLS[code]


# ---------------------------------------------------------------------------
# ncurrency — locale-aware currency formatter
# ---------------------------------------------------------------------------

def _format_single_currency(value: float, symbol: str, digits: int,
                            locale_obj: Optional[Locale],
                            unit: str, unit_labels: Optional[Dict[str, str]]) -> str:
    """Format a single numeric value as a currency string."""
    if not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "∞" if value > 0 else "-∞"
    
    # Resolve locale settings
    if locale_obj is not None:
        tsep = locale_obj.thousand_separator
        dsep = locale_obj.decimal_separator
        grouping = locale_obj.grouping
        pos = locale_obj.currency_position
        spacing = locale_obj.currency_spacing
        _unit_labels = unit_labels or locale_obj.unit_labels
        _unit_thresholds = locale_obj.unit_thresholds
    else:
        tsep = ","
        dsep = "."
        grouping = (3,)
        pos = "prefix"
        spacing = False
        _unit_labels = unit_labels or {
            "thousand": "K", "million": "Mn", "billion": "Bn", "trillion": "Tn"
        }
        _unit_thresholds = (
            (1e12, 1e-12, "trillion"),
            (1e9,  1e-9,  "billion"),
            (1e6,  1e-6,  "million"),
            (1e3,  1e-3,  "thousand"),
        )
    
    is_negative = value < 0
    abs_val = abs(value)
    
    # Determine unit scaling
    unit_suffix = ""
    if unit == 'custom':
        for threshold, factor, label_key in _unit_thresholds:
            if abs_val >= threshold:
                abs_val = abs_val * abs(factor)
                unit_suffix = " " + _unit_labels.get(label_key, "")
                break
    elif unit == '' or unit is None:
        pass  # No scaling
    else:
        # Fixed unit
        for threshold, factor, label_key in _unit_thresholds:
            if _unit_labels.get(label_key, "") == unit:
                abs_val = abs_val * abs(factor)
                unit_suffix = " " + unit
                break
    
    # Format the number
    # Split into integer and decimal parts
    fmt = f"{abs_val:.{digits}f}"
    
    if "." in fmt:
        int_part, dec_part = fmt.split(".")
    else:
        int_part = fmt
        dec_part = ""
    
    # Remove any existing commas (from f-string, shouldn't be any, but safe)
    int_part = int_part.replace(",", "")
    
    # Apply grouping
    grouped_int = format_grouped_number(int_part, grouping, tsep)
    
    # Recombine
    if dec_part:
        num_str = grouped_int + dsep + dec_part
    else:
        num_str = grouped_int
    
    # Add sign
    if is_negative:
        num_str = "-" + num_str
    
    # Add unit suffix
    num_str = num_str + unit_suffix
    
    # Combine with currency symbol
    space = " " if spacing else ""
    if pos == "prefix":
        return symbol + space + num_str
    else:
        return num_str + space + symbol


def ncurrency(number: "Union[np.ndarray, list, float, int]",
              currency: Optional[str] = None,
              digits: int = 2,
              unit: str = 'custom',
              unit_labels: Optional[Dict[str, str]] = None,
              locale: "Optional[str | Locale]" = None) -> "Union[np.ndarray, str]":
    """
    Format numbers as currency values with locale-aware formatting.
    
    Parameters
    ----------
    number : array-like
        Input numbers.
    currency : str, optional
        ISO 4217 currency code (e.g. "INR", "USD", "EUR").
        Defaults to the locale's default currency.
    digits : int, default 2
        Number of decimal digits to display.
    unit : str, default 'custom'
        'custom': Determine best unit for each number individually.
        '': No unit scaling — show full number.
        Or a specific unit label (e.g. 'K', 'Cr').
    unit_labels : dict, optional
        Override unit labels.
    locale : str or Locale, optional
        Locale for formatting. Falls back to global locale.
    
    Returns
    -------
    numpy.ndarray or str
        Formatted currency strings.
    
    Examples
    --------
    >>> ncurrency(1_50_00_000, currency="INR", locale="en_IN")
    '₹1.5 Cr'
    
    >>> ncurrency(1500000, currency="USD")
    '$1.5 Mn'
    """
    _check_singleton(digits, 'digits', int)
    
    locale_obj = resolve_locale(locale)
    
    # Resolve currency
    if currency is None:
        if locale_obj is not None:
            currency = locale_obj.default_currency
        else:
            currency = "USD"
    
    symbol = get_currency_symbol(currency)
    
    # Handle scalar input
    is_scalar = np.isscalar(number) and not isinstance(number, (np.ndarray, list))
    
    x_arr = np.asanyarray(number, dtype=float)
    original_shape = x_arr.shape
    x_flat = x_arr.ravel()
    
    # Optimize via unique values
    uvals, inverse = np.unique(x_flat, return_inverse=True)
    
    formatted_list = [
        _format_single_currency(v, symbol, digits, locale_obj, unit, unit_labels)
        for v in uvals
    ]
    formatted_arr = np.array(formatted_list, dtype=object)
    
    result = formatted_arr[inverse].reshape(original_shape)
    
    if is_scalar or result.ndim == 0:
        return result.item()
    
    return result
