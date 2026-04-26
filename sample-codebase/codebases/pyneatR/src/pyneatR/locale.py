"""
Locale system for pyneatR.

Provides locale-aware formatting defaults for numbers, dates, and currencies.
Supports Indian (en_IN), US (en_US), and EU (de_DE) locales out of the box.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Locale dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Locale:
    """
    Immutable definition of formatting conventions for a locale.
    
    Parameters
    ----------
    name : str
        Locale identifier (e.g. "en_IN", "en_US", "de_DE").
    thousand_separator : str
        Character used to group digits (e.g. "," or ".").
    decimal_separator : str
        Character used as the decimal point (e.g. "." or ",").
    grouping : tuple of int
        Digit grouping sizes, right-to-left.  (3,) for standard,
        (3, 2) for Indian (first group of 3, then groups of 2).
    unit_labels : dict
        Mapping of magnitude names to display labels.
    unit_thresholds : tuple of tuple
        Pairs of (threshold, factor, label_key) used when unit='custom'.
        Evaluated top-down; first matching threshold wins.
    default_currency : str
        ISO 4217 currency code (e.g. "INR", "USD", "EUR").
    currency_position : str
        "prefix" or "suffix" — where the currency symbol appears.
    currency_spacing : bool
        If True, insert a space between symbol and number.
    date_order : str
        "MDY" (US) or "DMY" (India / EU).
    timezone_label : str
        Default timezone abbreviation for timestamps (e.g. "IST", "EST").
    percent_space : bool
        If True, insert a space before the percent sign (EU convention).
    """
    name: str
    
    # Number formatting
    thousand_separator: str = ","
    decimal_separator: str = "."
    grouping: Tuple[int, ...] = (3,)
    
    # Unit labels & thresholds
    unit_labels: Dict[str, str] = field(default_factory=lambda: {
        "thousand": "K", "million": "Mn", "billion": "Bn", "trillion": "Tn"
    })
    unit_thresholds: Tuple[Tuple[float, float, str], ...] = (
        (1e12, 1e-12, "trillion"),
        (1e9,  1e-9,  "billion"),
        (1e6,  1e-6,  "million"),
        (1e3,  1e-3,  "thousand"),
    )
    
    # Currency
    default_currency: str = "USD"
    currency_position: str = "prefix"
    currency_spacing: bool = False
    
    # Dates
    date_order: str = "MDY"
    timezone_label: str = ""
    
    # Percent
    percent_space: bool = False


# ---------------------------------------------------------------------------
# Predefined locales
# ---------------------------------------------------------------------------

LOCALES: Dict[str, Locale] = {
    "en_IN": Locale(
        name="en_IN",
        thousand_separator=",",
        decimal_separator=".",
        grouping=(3, 2),
        unit_labels={
            "thousand": "K",
            "lakh": "L",
            "crore": "Cr",
            "arab": "Arab",
            "kharab": "Kharab",
        },
        unit_thresholds=(
            (1e11, 1e-11, "kharab"),
            (1e9,  1e-9,  "arab"),
            (1e7,  1e-7,  "crore"),
            (1e5,  1e-5,  "lakh"),
            (1e3,  1e-3,  "thousand"),
        ),
        default_currency="INR",
        currency_position="prefix",
        currency_spacing=False,
        date_order="DMY",
        timezone_label="IST",
        percent_space=False,
    ),
    "en_US": Locale(
        name="en_US",
        thousand_separator=",",
        decimal_separator=".",
        grouping=(3,),
        unit_labels={
            "thousand": "K",
            "million": "M",
            "billion": "B",
            "trillion": "T",
        },
        unit_thresholds=(
            (1e12, 1e-12, "trillion"),
            (1e9,  1e-9,  "billion"),
            (1e6,  1e-6,  "million"),
            (1e3,  1e-3,  "thousand"),
        ),
        default_currency="USD",
        currency_position="prefix",
        currency_spacing=False,
        date_order="MDY",
        timezone_label="EST",
        percent_space=False,
    ),
    "de_DE": Locale(
        name="de_DE",
        thousand_separator=".",
        decimal_separator=",",
        grouping=(3,),
        unit_labels={
            "thousand": "K",
            "million": "Mio",
            "billion": "Mrd",
            "trillion": "Bio",
        },
        unit_thresholds=(
            (1e12, 1e-12, "trillion"),
            (1e9,  1e-9,  "billion"),
            (1e6,  1e-6,  "million"),
            (1e3,  1e-3,  "thousand"),
        ),
        default_currency="EUR",
        currency_position="suffix",
        currency_spacing=True,
        date_order="DMY",
        timezone_label="CET",
        percent_space=True,
    ),
}


# ---------------------------------------------------------------------------
# Global locale state
# ---------------------------------------------------------------------------

_current_locale: Optional[Locale] = None


def set_locale(locale_name: str) -> None:
    """
    Set the global pyneatR locale.
    
    Parameters
    ----------
    locale_name : str
        One of the registered locale names (e.g. "en_IN", "en_US", "de_DE").
    
    Raises
    ------
    ValueError
        If the locale name is not registered.
    """
    global _current_locale
    if locale_name not in LOCALES:
        available = ", ".join(sorted(LOCALES.keys()))
        raise ValueError(
            f"Unknown locale '{locale_name}'. Available: {available}"
        )
    _current_locale = LOCALES[locale_name]


def get_locale() -> Optional[Locale]:
    """
    Get the current global locale, or None if not set.
    
    Returns
    -------
    Locale or None
    """
    return _current_locale


def reset_locale() -> None:
    """Reset the global locale to None (default behaviour)."""
    global _current_locale
    _current_locale = None


def resolve_locale(locale: "Optional[str | Locale]" = None) -> Optional[Locale]:
    """
    Resolve a locale argument to a Locale object.
    
    Priority: explicit argument > global locale > None.
    
    Parameters
    ----------
    locale : str or Locale or None
        If str, looked up from LOCALES registry.
        If Locale, used directly.
        If None, falls back to global locale.
    
    Returns
    -------
    Locale or None
    """
    if locale is None:
        return _current_locale
    if isinstance(locale, Locale):
        return locale
    if isinstance(locale, str):
        if locale not in LOCALES:
            available = ", ".join(sorted(LOCALES.keys()))
            raise ValueError(
                f"Unknown locale '{locale}'. Available: {available}"
            )
        return LOCALES[locale]
    raise TypeError(f"locale must be str or Locale, got {type(locale).__name__}")


# ---------------------------------------------------------------------------
# Indian grouping helper (used by numbers.py)
# ---------------------------------------------------------------------------

def format_grouped_number(integer_str: str, grouping: Tuple[int, ...],
                          separator: str) -> str:
    """
    Apply digit grouping to an integer string.
    
    Parameters
    ----------
    integer_str : str
        The integer part of a number (no sign, no decimal).
    grouping : tuple of int
        Group sizes, right to left.  (3,) = standard; (3, 2) = Indian.
    separator : str
        The thousands separator character.
    
    Returns
    -------
    str
        Grouped integer string.
    
    Examples
    --------
    >>> format_grouped_number("10000000", (3, 2), ",")
    '1,00,00,000'
    >>> format_grouped_number("10000000", (3,), ",")
    '10,000,000'
    """
    if len(integer_str) == 0:
        return "0"
    
    # Process groups from the right
    result_parts = []
    remaining = integer_str
    
    for i, size in enumerate(grouping):
        if len(remaining) <= size:
            break
        result_parts.append(remaining[-size:])
        remaining = remaining[:-size]
        # If this is the last group size specified, repeat it
        if i == len(grouping) - 1:
            while len(remaining) > size:
                result_parts.append(remaining[-size:])
                remaining = remaining[:-size]
            break
    
    # Whatever is left goes as the leading group
    if remaining:
        result_parts.append(remaining)
    
    result_parts.reverse()
    return separator.join(result_parts)
