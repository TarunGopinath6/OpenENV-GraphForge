"""
DataFrame auto-formatting hooks for pyneatR.

Provides `neat_pandas()`, `neat_polars()`, `activate()`, and `deactivate()`
for automatic pyneatR formatting of DataFrame display output.
"""
import numpy as np
from typing import Optional, Dict, Callable, Any

from .numbers import nnumber, npercent
from .currency import ncurrency
from .dates import ndate, ntimestamp
from .strings import nstring
from .locale import resolve_locale, set_locale


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------

# Column name patterns that hint at specific types
_PERCENT_HINTS = {"pct", "percent", "rate", "ratio", "growth", "churn",
                  "conversion", "yield", "margin", "share"}
_CURRENCY_HINTS = {"price", "amount", "cost", "revenue", "fee", "salary",
                   "income", "expense", "total", "balance", "payment",
                   "budget", "profit", "loss", "tax", "discount"}
_DATE_HINTS = {"date", "dt", "day", "dob", "birth_date", "start_date",
               "end_date", "created_date", "updated_date"}
_TIMESTAMP_HINTS = {"timestamp", "ts", "created_at", "updated_at",
                    "modified_at", "event_time", "log_time", "datetime"}
_ID_HINTS = {"id", "idx", "index", "key", "code", "pk", "fk", "uid",
             "uuid", "hash"}


def _column_name_matches(col_name: str, hints: set) -> bool:
    """Check if a column name matches any hint pattern."""
    col_lower = col_name.lower().strip()
    
    # Exact match
    if col_lower in hints:
        return True
    
    # Suffix/prefix match (e.g. "user_id", "order_date")
    for hint in hints:
        if col_lower.endswith("_" + hint) or col_lower.startswith(hint + "_"):
            return True
        if col_lower.endswith(hint):
            return True
    
    return False


def _infer_column_type_pandas(series: "Any") -> str:
    """
    Infer pyneatR format type for a Pandas Series.
    
    Returns one of: 'number', 'percent', 'currency', 'date', 'timestamp',
    'string', 'skip'.
    """
    import pandas as pd
    
    col_name = str(series.name) if series.name is not None else ""
    
    # Skip ID-like columns
    if _column_name_matches(col_name, _ID_HINTS):
        return 'skip'
    
    # Check dtype first (fast path)
    if pd.api.types.is_datetime64_any_dtype(series):
        # Check if all times are midnight → likely date-only
        non_null = series.dropna()
        if len(non_null) > 0:
            times = non_null.dt.time
            if all(t.hour == 0 and t.minute == 0 and t.second == 0 for t in times):
                return 'date'
        return 'timestamp'
    
    if pd.api.types.is_numeric_dtype(series):
        # Check column name hints
        if _column_name_matches(col_name, _PERCENT_HINTS):
            return 'percent'
        if _column_name_matches(col_name, _CURRENCY_HINTS):
            return 'currency'
        
        # Heuristic: values in [0, 1] range → likely ratio/percent
        non_null = series.dropna()
        if len(non_null) > 0:
            if non_null.min() >= -1 and non_null.max() <= 1:
                # Could be a ratio, but only if not all integers
                if not all(v == int(v) for v in non_null if np.isfinite(v)):
                    return 'percent'
        
        return 'number'
    
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        # Check if it could be date strings
        if _column_name_matches(col_name, _DATE_HINTS):
            return 'date'
        if _column_name_matches(col_name, _TIMESTAMP_HINTS):
            return 'timestamp'
        return 'string'
    
    return 'skip'


def _infer_column_type_polars(series: "Any") -> str:
    """
    Infer pyneatR format type for a Polars Series.
    
    Returns one of: 'number', 'percent', 'currency', 'date', 'timestamp',
    'string', 'skip'.
    """
    import polars as pl
    
    col_name = series.name or ""
    dtype = series.dtype
    
    # Skip ID-like columns
    if _column_name_matches(col_name, _ID_HINTS):
        return 'skip'
    
    if dtype == pl.Date:
        return 'date'
    
    if dtype in (pl.Datetime, pl.Time):
        return 'timestamp'
    
    if dtype.is_numeric():
        if _column_name_matches(col_name, _PERCENT_HINTS):
            return 'percent'
        if _column_name_matches(col_name, _CURRENCY_HINTS):
            return 'currency'
        
        # Heuristic: values in [0, 1] range
        non_null = series.drop_nulls()
        if len(non_null) > 0:
            if non_null.min() >= -1 and non_null.max() <= 1:
                if dtype.is_float():
                    return 'percent'
        
        return 'number'
    
    if dtype in (pl.Utf8, pl.String):
        if _column_name_matches(col_name, _DATE_HINTS):
            return 'date'
        if _column_name_matches(col_name, _TIMESTAMP_HINTS):
            return 'timestamp'
        return 'string'
    
    return 'skip'


# ---------------------------------------------------------------------------
# Formatting helpers (scalar formatters for .style.format)
# ---------------------------------------------------------------------------

def _make_scalar_formatter(col_type: str, currency_code: Optional[str],
                           locale_str: Optional[str]) -> Optional[Callable]:
    """Create a scalar formatter function for a given column type."""
    
    if col_type == 'number':
        def fmt_number(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            return nnumber(x, locale=locale_str)
        return fmt_number
    
    elif col_type == 'percent':
        def fmt_percent(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            return npercent(x, is_ratio=True, digits=1, show_plus_sign=True)
        return fmt_percent
    
    elif col_type == 'currency':
        def fmt_currency(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            return ncurrency(x, currency=currency_code, locale=locale_str)
        return fmt_currency
    
    elif col_type == 'date':
        def fmt_date(x):
            if x is None:
                return "—"
            try:
                return ndate(x, show_weekday=False)
            except (ValueError, TypeError):
                return str(x)
        return fmt_date
    
    elif col_type == 'timestamp':
        def fmt_ts(x):
            if x is None:
                return "—"
            try:
                return ntimestamp(x, show_weekday=True)
            except (ValueError, TypeError):
                return str(x)
        return fmt_ts
    
    elif col_type == 'string':
        def fmt_str(x):
            if x is None:
                return ""
            return str(x)
        return fmt_str
    
    return None


# ---------------------------------------------------------------------------
# Pandas hook
# ---------------------------------------------------------------------------

def neat_pandas(df: "Any",
                column_types: Optional[Dict[str, str]] = None,
                locale: Optional[str] = None,
                currency: Optional[str] = None,
                infer: bool = True) -> "Any":
    """
    Apply pyneatR formatting to a Pandas DataFrame for display.
    
    Returns a Pandas Styler object with formatted display. The underlying
    data is NOT modified.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column_types : dict, optional
        Explicit mapping of column names to pyneatR format types.
        e.g. {"revenue": "currency", "growth": "percent"}
        Types: 'number', 'percent', 'currency', 'date', 'timestamp',
        'string', 'skip'.
    locale : str, optional
        Locale override (defaults to global locale).
    currency : str, optional
        Currency code override for currency-typed columns.
    infer : bool, default True
        If True, auto-infer column types for columns not in column_types.
    
    Returns
    -------
    pd.io.formats.style.Styler
        Styled DataFrame.
    """
    import pandas as pd
    
    locale_obj = resolve_locale(locale)
    if locale_obj is not None and currency is None:
        currency = locale_obj.default_currency
    
    format_dict = {}
    col_types = column_types or {}
    
    for col in df.columns:
        # Determine type
        col_type = col_types.get(col)
        if col_type is None and infer:
            col_type = _infer_column_type_pandas(df[col])
        
        if col_type is None or col_type == 'skip':
            continue
        
        formatter = _make_scalar_formatter(col_type, currency, locale)
        if formatter is not None:
            format_dict[col] = formatter
    
    return df.style.format(format_dict, na_rep="—")


# ---------------------------------------------------------------------------
# Polars hook
# ---------------------------------------------------------------------------

def neat_polars(df: "Any",
                column_types: Optional[Dict[str, str]] = None,
                locale: Optional[str] = None,
                currency: Optional[str] = None,
                infer: bool = True) -> "Any":
    """
    Apply pyneatR formatting to a Polars DataFrame for display.
    
    Returns a Great Tables GT object for rich HTML rendering.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    column_types : dict, optional
        Explicit mapping of column names to pyneatR format types.
    locale : str, optional
        Locale override.
    currency : str, optional
        Currency code override for currency-typed columns.
    infer : bool, default True
        If True, auto-infer column types.
    
    Returns
    -------
    great_tables.GT
        Styled table object.
    """
    import polars as pl
    import great_tables as gt
    
    locale_obj = resolve_locale(locale)
    if locale_obj is not None and currency is None:
        currency = locale_obj.default_currency
    
    col_types = column_types or {}
    
    tbl = gt.GT(df)
    
    for col in df.columns:
        col_type = col_types.get(col)
        if col_type is None and infer:
            col_type = _infer_column_type_polars(df[col])
        
        if col_type is None or col_type == 'skip':
            continue
        
        formatter = _make_scalar_formatter(col_type, currency, locale)
        if formatter is not None:
            tbl = tbl.fmt(columns=col, fns=formatter)
    
    return tbl


# ---------------------------------------------------------------------------
# activate() / deactivate()
# ---------------------------------------------------------------------------

# Store original repr methods for deactivation
_original_pandas_repr = None
_original_polars_formatter = None
_activated = False


def activate(locale: Optional[str] = None,
             currency: Optional[str] = None) -> None:
    """
    Activate pyneatR auto-formatting for DataFrame display.
    
    When activated, Pandas and Polars DataFrames will automatically
    use pyneatR formatting when displayed in Jupyter notebooks.
    
    Parameters
    ----------
    locale : str, optional
        Locale to set globally (e.g. "en_IN", "en_US", "de_DE").
    currency : str, optional
        Default currency code for currency-typed columns.
    
    Notes
    -----
    This modifies global state by patching ``pd.DataFrame._repr_html_``
    and registering an IPython display formatter for Polars DataFrames.
    Call ``deactivate()`` to restore original behavior.
    """
    global _original_pandas_repr, _original_polars_formatter, _activated
    
    if locale:
        set_locale(locale)
    
    # — Pandas hook —
    try:
        import pandas as pd
        
        _original_pandas_repr = pd.DataFrame._repr_html_
        
        _currency = currency  # capture for closure
        _locale = locale
        
        def _neat_repr_html(self):
            try:
                styled = neat_pandas(self, locale=_locale, currency=_currency)
                return styled.to_html()
            except Exception:
                # Fallback to original repr if formatting fails
                if _original_pandas_repr is not None:
                    return _original_pandas_repr(self)
                return repr(self)
        
        pd.DataFrame._repr_html_ = _neat_repr_html
        
    except ImportError:
        pass
    
    # — Polars hook (via IPython) —
    try:
        import polars as pl
        
        ip = get_ipython()  # noqa: F821 — only available in IPython/Jupyter
        formatter = ip.display_formatter.formatters['text/html']
        
        _currency_pl = currency
        _locale_pl = locale
        
        def _neat_polars_html(df):
            try:
                gt_obj = neat_polars(df, locale=_locale_pl, currency=_currency_pl)
                return gt_obj.as_raw_html()
            except Exception:
                return None  # Fall back to default repr
        
        _original_polars_formatter = formatter.for_type(
            pl.DataFrame, _neat_polars_html
        )
        
    except (ImportError, NameError):
        # NameError if get_ipython() not available (not in Jupyter)
        pass
    
    _activated = True


def deactivate() -> None:
    """
    Deactivate pyneatR auto-formatting and restore original DataFrame display.
    """
    global _original_pandas_repr, _original_polars_formatter, _activated
    
    if not _activated:
        return
    
    # Restore Pandas
    try:
        import pandas as pd
        if _original_pandas_repr is not None:
            pd.DataFrame._repr_html_ = _original_pandas_repr
            _original_pandas_repr = None
    except ImportError:
        pass
    
    # Restore Polars
    try:
        import polars as pl
        ip = get_ipython()  # noqa: F821
        formatter = ip.display_formatter.formatters['text/html']
        
        if _original_polars_formatter is not None:
            formatter.for_type(pl.DataFrame, _original_polars_formatter)
        else:
            formatter.pop(pl.DataFrame, None)
        
        _original_polars_formatter = None
            
    except (ImportError, NameError):
        pass
    
    _activated = False


def is_activated() -> bool:
    """Check if pyneatR auto-formatting is currently active."""
    return _activated
