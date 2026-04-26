import numpy as np
import datetime
from .utils import _check_singleton, _unique_optimization
from typing import Union

def _get_weekday_name_vec(dt64: np.ndarray) -> np.ndarray:
    """
    Vectorized weekday name extraction.

    Parameters
    ----------
    dt64 : numpy.ndarray
        Array of datetime64 values.

    Returns
    -------
    numpy.ndarray
        Array of weekday abbreviations.
    """
    days = dt64.astype('datetime64[D]').astype(int)
    idx = (days + 3) % 7
    labels = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    return labels[idx]

@_unique_optimization
def nday(date: Union[np.ndarray, list, datetime.date], show_relative_day: bool = False) -> Union[np.ndarray, str]:
    """
    Format dates as day names, optionally with relative alias (Today, Yesterday, etc).

    Parameters
    ----------
    date : array-like
        Input date(s).
    show_relative_day : bool, default False
        If True, adds context like 'Today', 'Yesterday', 'Coming', 'Last'.

    Returns
    -------
    numpy.ndarray or str
        Formatted day names or strings.
    """
    _check_singleton(show_relative_day, 'show_relative_day', bool)
    
    if np.size(date) == 0:
        return np.array([], dtype=object)
    
    if not np.issubdtype(getattr(date, 'dtype', None), np.datetime64):
         try:
             date = np.asanyarray(date).astype('datetime64[D]')
         except (ValueError, TypeError):
             pass

    if not isinstance(date, np.ndarray):
         date = np.asanyarray(date)
         
    if not np.issubdtype(date.dtype, np.datetime64):
         return np.full(date.shape, "NaT", dtype=object)

    mask = ~np.isnat(date)
    result = np.full(date.shape, "NaT", dtype=object)
    
    if not np.any(mask):
        return result
        
    valid_dates = date[mask]
    
    day_str = _get_weekday_name_vec(valid_dates)
    
    if show_relative_day:
        today = np.datetime64('today')
        d_day = valid_dates.astype('datetime64[D]')
        today_day = today.astype('datetime64[D]')
        
        diff = (today_day - d_day) / np.timedelta64(1, 'D')
        diff = diff.astype(int)
        
        alias = np.full(len(diff), "", dtype='<U20')
        
        alias[(diff >= 2) & (diff <= 8)] = "Last "
        alias[diff == 1] = "Yesterday, "
        alias[diff == 0] = "Today, "
        alias[diff == -1] = "Tomorrow, "
        alias[(diff >= -8) & (diff <= -2)] = "Coming "
        
        day_str = np.char.add(alias, day_str)
        
    result[mask] = day_str
    return result

@_unique_optimization
def ndate(date: Union[np.ndarray, list, datetime.date], show_weekday: bool = True, show_month_year: bool = False) -> Union[np.ndarray, str]:
    """
    Format dates into a neat string representation.

    Parameters
    ----------
    date : array-like
        Input date(s).
    show_weekday : bool, default True
        If True, appending weekday name e.g. (Mon).
    show_month_year : bool, default False
        If True, formats as month abbreviation and year (Jan'23).

    Returns
    -------
    numpy.ndarray or str
        Formatted date strings.
    """  
    _check_singleton(show_weekday, 'show_weekday', bool)
    _check_singleton(show_month_year, 'show_month_year', bool)
    
    if np.size(date) == 0:
        return np.array([], dtype=object)
    
    if not np.issubdtype(getattr(date, 'dtype', None), np.datetime64):
         try:
             date = np.asanyarray(date).astype('datetime64[D]')
         except (ValueError, TypeError):
             # Fallback attempt if astype fails or if it's already object
             date = np.asanyarray(date)

    if not isinstance(date, np.ndarray):
        date = np.asanyarray(date)
        
    if not np.issubdtype(date.dtype, np.datetime64):
        # Conversion failed, treat all as NaT/Invalid
        return np.full(date.shape, "NaT", dtype=object)

    mask = ~np.isnat(date)
    result = np.full(date.shape, "NaT", dtype=object)
    
    if not np.any(mask):
        return result
        
    valid_dates = date[mask]
    
    iso = np.datetime_as_string(valid_dates, unit='D')
    
    if show_month_year:
        months_full = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_arr = np.array(months_full)
        
        mm = np.array([s[5:7] for s in iso], dtype=int)
        yy = np.array([s[2:4] for s in iso])
        
        mon_str = months_arr[mm]
        
        s = np.char.add(np.char.add(mon_str, "'"), yy)
        
    else:
        months_full = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_arr = np.array(months_full)
        
        yyyy = np.array([s[0:4] for s in iso])
        mm = np.array([s[5:7] for s in iso], dtype=int)
        dd = np.array([s[8:10] for s in iso])
        
        mon_str = months_arr[mm]
        
        p1 = np.char.add(mon_str, " ")
        p2 = np.char.add(p1, dd)
        p3 = np.char.add(p2, ", ")
        s = np.char.add(p3, yyyy)
        
        if show_weekday:
            wd = _get_weekday_name_vec(valid_dates)
            w_part = np.char.add(" (", np.char.add(wd, ")"))
            s = np.char.add(s, w_part)
            
    result[mask] = s
    return result

@_unique_optimization
def ntimestamp(timestamp: Union[np.ndarray, list, datetime.datetime], 
               show_weekday: bool = True, show_date: bool = True,
               show_hours: bool = True, show_minutes: bool = True, show_seconds: bool = True,
               show_timezone: bool = True) -> Union[np.ndarray, str]:
    """
    Format timestamps into a neat string representation.

    Parameters
    ----------
    timestamp : array-like
        Input timestamp(s).
    show_weekday : bool, default True
        If True, appending weekday name e.g. (Mon).
    show_date : bool, default True
        If True, include date part.
    show_hours : bool, default True
        If True, include hours (12H format).
    show_minutes : bool, default True
        If True, include minutes.
    show_seconds : bool, default True
        If True, include seconds.
    show_timezone : bool, default True
        Reserved parameter for future timezone support (currently unused).

    Returns
    -------
    numpy.ndarray or str
        Formatted timestamp strings.
    """
    if np.size(timestamp) == 0:
        return np.array([], dtype=object)

    if not isinstance(timestamp, np.ndarray):
         timestamp = np.asanyarray(timestamp)

    if not np.issubdtype(timestamp.dtype, np.datetime64):
         try:
             timestamp = timestamp.astype('datetime64[s]')
         except (ValueError, TypeError):
             pass

    if not np.issubdtype(timestamp.dtype, np.datetime64):
         return np.full(timestamp.shape, "NaT", dtype=object)

    mask = ~np.isnat(timestamp)
    result = np.full(timestamp.shape, "NaT", dtype=object)
    
    if not np.any(mask):
        return result
        
    valid_ts = timestamp[mask]
    
    iso = np.datetime_as_string(valid_ts, unit='s')
    
    parts_list = []
    
    if show_date:
        months_full = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_arr = np.array(months_full)
        
        yyyy = np.array([s[0:4] for s in iso])
        mm = np.array([s[5:7] for s in iso], dtype=int)
        dd = np.array([s[8:10] for s in iso])
        mon_str = months_arr[mm]
        
        d_str = np.char.add(mon_str, " ")
        d_str = np.char.add(d_str, dd)
        d_str = np.char.add(d_str, ", ")
        d_str = np.char.add(d_str, yyyy)
        d_str = np.char.add(d_str, " ")
        
        parts_list.append(d_str)

    hh_str = np.array([s[11:13] for s in iso])
    mm_str = np.array([s[14:16] for s in iso])
    ss_str = np.array([s[17:19] for s in iso])
    
    hh_int = hh_str.astype(int)
    is_pm = hh_int >= 12
    
    hh_12 = hh_int.copy()
    hh_12[hh_12 == 0] = 12
    hh_12[hh_12 > 12] -= 12
    
    hours_labels = np.array([f"{i:02d}" for i in range(13)])
    hh_12_str = hours_labels[hh_12]
    
    if show_hours:
        parts_list.append(np.char.add(hh_12_str, "H"))
        
    if show_minutes:
        parts_list.append(np.char.add(" ", np.char.add(mm_str, "M")))
        
    if show_seconds:
        parts_list.append(np.char.add(" ", np.char.add(ss_str, "S")))
        
    if show_timezone:
        pass
        
    ampm_arr = np.where(is_pm, " PM", " AM")
    parts_list.append(ampm_arr)
    
    combined = parts_list[0]
    for p in parts_list[1:]:
        combined = np.char.add(combined, p)
        
    if show_weekday:
        wd = _get_weekday_name_vec(valid_ts)
        w_part = np.char.add(" (", np.char.add(wd, ")"))
        combined = np.char.add(combined, w_part)
        
    result[mask] = combined
    return result
