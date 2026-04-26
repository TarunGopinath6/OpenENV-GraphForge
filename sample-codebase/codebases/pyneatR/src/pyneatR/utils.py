import numpy as np
import functools
from typing import Any, Optional, Type, Callable

def _to_numpy(x: Any) -> np.ndarray:
    """
    Convert input to a numpy array.
    
    Parameters
    ----------
    x : Any
        Input data (list, tuple, array, series).

    Returns
    -------
    numpy.ndarray
        Numpy array representation of input.
    """
    return np.asanyarray(x)

def _check_singleton(x: Any, var_name: str, type_check: Optional[Type] = None) -> None:
    """
    Check if x is a singleton and optionally of a specific type.
    
    Parameters
    ----------
    x : Any
        Value to check.
    var_name : str
        Name of the variable for error message.
    type_check : Type, optional
        Expected type (e.g. int, str).

    Raises
    ------
    ValueError
        If x is not a singleton.
    TypeError
        If x is not of expected type.
    """
    if np.ndim(x) != 0 and np.size(x) != 1:
         raise ValueError(f"`{var_name}` must be a singleton.")
    
    if type_check:
        if not isinstance(x, type_check):
            if type_check is bool and isinstance(x, (bool, np.bool_)):
                return
            if type_check is str and isinstance(x, (str, np.str_)):
                return
            if type_check is int and isinstance(x, (int, np.integer)):
                return
            if type_check is float and isinstance(x, (float, np.floating)):
                return
            raise TypeError(f"`{var_name}` must be of type {type_check.__name__}.")

def _sandwich(x: np.ndarray, prefix: str = "", suffix: str = "") -> np.ndarray:
    """
    Add prefix and suffix to strings in x.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of strings.
    prefix : str, default ''
        String to prepend.
    suffix : str, default ''
        String to append.

    Returns
    -------
    numpy.ndarray
        Modified string array.
    """
    if prefix:
        x = np.char.add(prefix, x)
    if suffix:
        x = np.char.add(x, suffix)
        
    return x

def _unique_optimization(func: Callable) -> Callable:
    """
    Decorator to apply the unique value optimization pattern.
    
    Optimizes vectorized functions by computing results only on unique values
    and mapping them back to the original array shape.
    
    Parameters
    ----------
    func : Callable
        Function to wrap.

    Returns
    -------
    Callable
        Optimized function.
    """
    @functools.wraps(func)
    def wrapper(x, **kwargs):
        x_arr = _to_numpy(x)
        original_shape = x_arr.shape
        x_flat = x_arr.ravel()
        
        try:
            uvals, inverse = np.unique(x_flat, return_inverse=True)
        except (TypeError, ValueError):
            # Fallback for mixed types or other unique failures
            uvals = x_flat
            inverse = np.arange(len(x_flat))
        
        formatted_uvals = func(uvals, **kwargs)
        
        result_flat = formatted_uvals[inverse]
        result = result_flat.reshape(original_shape)
        
        if result.ndim == 0:
            return result.item()
            
        return result
    
    return wrapper
