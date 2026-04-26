import numpy as np
import datetime
from .dates import ndate, ntimestamp, nday
from .numbers import nnumber, npercent
from .strings import nstring
from .currency import ncurrency
from .locale import resolve_locale
from typing import Union, Any, Optional

def _infer_type(x: Any) -> str:
    """
    Infer the format type from the input.
    """
    if isinstance(x, (datetime.datetime, np.datetime64)):
        # Check if it's a date or timestamp
        if isinstance(x, datetime.datetime):
            return 'ts'
        # For numpy datetime64, check unit
        if "datetime64[D]" in str(getattr(x, 'dtype', '')):
            return 'date'
        return 'ts'
    
    if isinstance(x, (datetime.date)):
        return 'date'
        
    x_arr = np.asanyarray(x)
    
    if np.issubdtype(x_arr.dtype, np.datetime64):
         # If unit is D, it's date, else ts
         if "[D]" in str(x_arr.dtype):
             return 'date'
         return 'ts'
         
    if np.issubdtype(x_arr.dtype, np.number):
        return 'number'
        
    if np.issubdtype(x_arr.dtype, np.object_):
        # Check if it's mixed numeric
        try:
            temp = x_arr.astype(float)
            if not np.all(np.isnan(temp)):
                return 'number'
        except (ValueError, TypeError):
            pass
        return 'string'
        
    return 'string'

def f(x: Any, format_type: Optional[str] = None, locale: "Optional[str]" = None,
      **kwargs) -> Union[np.ndarray, str]:
    """
    Smart format function that infers type and applies pyneatR formatting.
    
    Parameters
    ----------
    x : Any
        Input data to format.
    format_type : str, optional
        Explicit format type: 'day', 'date', 'ts', 'number', 'percent',
        'currency', 'string'.
        If None, type is inferred.
    locale : str, optional
        Locale name (e.g. "en_IN", "en_US", "de_DE").
        Overrides the global locale for this call.
    **kwargs : dict
        Additional parameters passed to the underlying formatting functions.
    """
    locale_obj = resolve_locale(locale)
    
    if format_type is None:
        format_type = _infer_type(x)
        
    if format_type == 'day':
        # Default: Mon, Tue, ...
        params = {'show_relative_day': False}
        params.update(kwargs)
        return nday(x, **params)
        
    elif format_type == 'date':
        # Default: Jan 01, 2026
        params = {'show_weekday': False}
        params.update(kwargs)
        return ndate(x, **params)
        
    elif format_type == 'ts':
        # Timezone label comes from locale, not hardcoded
        tz_label = ""
        if locale_obj is not None:
            tz_label = locale_obj.timezone_label
        # Allow explicit override via kwargs
        tz_label = kwargs.pop('timezone_label', tz_label)
        
        params = {'show_weekday': True, 'show_timezone': True}
        params.update(kwargs)
        res = ntimestamp(x, **params)
        
        # Add timezone label before the weekday part if available
        if tz_label and params.get('show_timezone', True):
            if isinstance(res, np.ndarray):
                def inject_tz(s):
                    idx = s.rfind(" (")
                    if idx != -1:
                        return s[:idx] + " " + tz_label + s[idx:]
                    return s + " " + tz_label
                    
                inject_vec = np.vectorize(inject_tz, otypes=[object])
                res = inject_vec(res)
            else:
                idx = res.rfind(" (")
                if idx != -1:
                    res = res[:idx] + " " + tz_label + res[idx:]
                else:
                    res += " " + tz_label
        return res
        
    elif format_type == 'number':
        # Default: 1,345 Bn (K, Mn, Bn, Tn, comma separator)
        params = {'thousand_separator': ',', 'unit': 'custom'}
        if locale is not None:
            params['locale'] = locale
        params.update(kwargs)
        return nnumber(x, **params)
    
    elif format_type == 'currency':
        # Currency formatting — opt-in via format_type='currency'
        params = {}
        if locale is not None:
            params['locale'] = locale
        params.update(kwargs)
        return ncurrency(x, **params)
        
    elif format_type == 'percent':
        # Default: +900% (9x growth, 90K basis points)
        # We call npercent with custom logic to match the requested format
        is_ratio = kwargs.get('is_ratio', True)
        digits = kwargs.get('digits', 1)
        show_plus_sign = kwargs.get('show_plus_sign', True)
        
        x_val = np.asanyarray(x, dtype=float)
        
        # 1. Main percentage string
        main_pct = npercent(x, is_ratio=is_ratio, digits=digits, show_plus_sign=show_plus_sign, 
                            show_growth_factor=False, show_bps=False)
        
        # 2. Growth factor string (e.g. 9x growth) - remove space
        mult = x_val if is_ratio else x_val / 100.0
        growth_val = np.abs(mult)
        # Format growth: no space between number and x
        growth_str_list = [f"{v:.1f}" if v % 1 != 0 else f"{int(v)}" for v in np.atleast_1d(growth_val)]
        growth_labels = np.where(mult >= 0, "x growth", "x drop")
        growth_labels_1d = np.atleast_1d(growth_labels)
        growth_full = np.array([f"{s}{lbl}" for s, lbl in zip(growth_str_list, growth_labels_1d)])
        
        # 3. Basis points string (e.g. 90K basis points) - remove space in nnumber output
        bps_val = x_val * 10000 if is_ratio else x_val * 100
        bps_fmt_arr = nnumber(bps_val, digits=0, thousand_separator=',')
        # Remove space before K, Mn, Bn, etc. in bps
        bps_fmt_clean = np.char.replace(bps_fmt_arr, " ", "")
        bps_full = np.char.add(bps_fmt_clean, " basis points")
        
        # Combine: +900% (9x growth, 90K basis points)
        rhs = np.char.add(" (", np.char.add(growth_full, np.char.add(", ", np.char.add(bps_full, ")"))))
        comp = np.char.add(main_pct, rhs)
        
        if np.isscalar(x) and not isinstance(x, (np.ndarray, list)):
             return comp[0]
        return comp

    elif format_type == 'string':
        # Default: ascii_only, title case, remove special characters
        params = {'case': 'title', 'remove_specials': True, 'ascii_only': True}
        params.update(kwargs)
        return nstring(x, **params)
        
    else:
        raise ValueError(f"Unknown format_type: {format_type}")
