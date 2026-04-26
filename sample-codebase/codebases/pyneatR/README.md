# pyneatR

Neat formatting for numbers, dates, timestamps, strings, and currencies using numpy.
Implementation inspired by the R package [`neatR`](https://cran.r-project.org/web/packages/neatR/vignettes/neat-data.html).

## Installation

```bash
pip install pyneatR
```

## Features

- **Numbers** — auto-scaling with K, Mn, Bn, Tn units
- **Percentages** — with growth factors and basis points
- **Dates** — clean `mmm dd, yyyy` format with weekday context
- **Timestamps** — 12H format with AM/PM and timezone labels
- **Strings** — case conversion and special character removal
- **Currency** — 60+ Unicode symbols (₹ $ € £ ¥ ₿ …) with locale-aware formatting
- **Locale** — Indian (Lakhs/Crores), US, and EU formatting out of the box
- **DataFrame hooks** — auto-format Pandas and Polars DataFrames for display

---

## Neat Data for Presentation

### Formatting dates

Often, we encounter dates in various formats and want an unambiguous representation. `ndate` formats dates into a readable `mmm dd, yyyy` format with the day of the week.

```python
from pyneatR import ndate, nday
from datetime import date, timedelta

today = date.today()
# Assuming today is Nov 09, 2025 (Sun) for these examples

print(ndate(today - timedelta(days=3)))
# "Nov 06, 2025 (Thu)"

print(ndate(today))
# "Nov 09, 2025 (Sun)"

print(ndate(today + timedelta(days=4)))
# "Nov 13, 2025 (Thu)"
```

To just get the date without the day of week, set `show_weekday=False`:

```python
print(ndate(today, show_weekday=False))
# "Nov 09, 2025"
```

For monthly data, abbreviating the date to `mmm'yy` is elegant:

```python
print(ndate(today, show_weekday=False, show_month_year=True))
# "Nov'25"
```

To see the context of the date with respect to current date (within 1 week), use `nday`.

```python
print(nday(today, show_relative_day=False))
# "Sun"

print(nday(today, show_relative_day=True))
# "Today, Sun"
```

### Formatting timestamp

Timestamps are feature-rich representations of date and time.

```python
from pyneatR import ntimestamp
from datetime import datetime

now = datetime.now()
# Assuming now is Nov 09, 2025 12:07:48 PM

print(ntimestamp(now))
# "Nov 09, 2025 12H 07M 48S PM (Sun)"
```

To extract and format only the time:

```python
print(ntimestamp(now, show_weekday=False, show_date=False, show_timezone=False))
# "12H 07M 48S PM"
```

Components of time can be toggled on or off:

```python
print(ntimestamp(now, show_date=False, show_weekday=False, 
                 show_hours=True, show_minutes=True, show_seconds=False, 
                 show_timezone=False))
# "12H 07M PM"
```

### Formatting number

`nnumber` formats numeric data in a readable way, automatically handling large numbers.

```python
from pyneatR import nnumber
import numpy as np

x = np.array([10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000])

print(nnumber(x))
# ['10.0' '100.0' '1.0 K' '10.0 K' '100.0 K' '1.0 Mn' '10.0 Mn' '100.0 Mn' '1.0 Bn']

print(nnumber(x, digits=0))
# ['10' '100' '1 K' '10 K' '100 K' '1 Mn' '10 Mn' '100 Mn' '1 Bn']
```

`nnumber` can automatically determine the best single unit for all numbers using `unit='auto'`:

```python
x = np.array([1e6, 99e3, 76e3, 42e3, 12e3, 789, 53])
print(nnumber(x, unit='auto'))
# ['1,000 K' '99 K' '76 K' '42 K' '12 K' '0.8 K' '0.1 K']
```

You can specify the unit directly:

```python
print(nnumber(123456789.123456, unit='Mn'))
# "123.5 Mn"
```

Default units are 'K', 'Mn', 'Bn', 'Tn'. Custom labels can be provided:

```python
print(nnumber(123456789.123456, unit='M', unit_labels={'million': 'M'}))
# "123.5 M"
```

Prefixes and suffixes can be added:

```python
print(nnumber(123456789.123456, unit='Mn', prefix='$ '))
# "$ 123.5 Mn"
```

To show the raw number with separators:

```python
print(nnumber(123456789.123456, digits=2, unit='', thousand_separator=','))
# "123,456,789.12"
```

### Formatting percentages

`npercent` formats percentages, handling both decimals (0.228) and values already multiplied by 100 (22.8).

```python
from pyneatR import npercent

print(npercent(22.8, is_ratio=False))
# "+22.8%"

print(npercent(0.228, is_ratio=True))
# "+22.8%"
```

By default `is_ratio=True`. The plus sign can be toggled:

```python
print(npercent(0.228, show_plus_sign=False))
# "22.8%"
```

For growth factors:

```python
tesla_2017 = 20
tesla_2023 = 200
g = (tesla_2023 - tesla_2017) / tesla_2017

print(npercent(g, show_plus_sign=True, show_growth_factor=True))
# "+900.0% (9.0x Growth)"
```

### Formatting string

`nstring` formats character vectors, removing special characters or changing case.

```python
from pyneatR import nstring

s = ' All MOdels are wrong. some ARE useful!!! '
print(nstring(s, case='title', remove_specials=True))
# "All Models Are Wrong Some Are Useful"
```

To retain only English alphabets and numbers:

```python
print(nstring(s, case='title', remove_specials=True, ascii_only=True))
# "All Models Are Wrong Some Are Useful"
```

### Smart Formatting with `f`

The `f` function is a "smart" formatter that infers the input type and applies the most appropriate `pyneatR` formatting with standardized defaults.

```python
from pyneatR import f
from datetime import date, datetime

# Infers Date
print(f(date(2026, 1, 1)))
# "Jan 01, 2026"

# Infers Timestamp (with Weekday)
print(f(datetime(2025, 11, 9, 12, 7, 48)))
# "Nov 09, 2025 12H 07M 48S PM (Sun)"

# Infers Number (with scaling and separators)
print(f(1345000000000))
# "1.3 Tn"

# Infers Percent (with growth factor and bps)
print(f(9.0, format_type='percent'))
# "+900.0% (9x growth, 90K basis points)"

# Infers String (Title Case + Cleaning)
print(f("all Models are Wrong!!!"))
# "All Models Are Wrong"
```

You can explicitly set `format_type` if inference isn't what you want: `day`, `date`, `ts`, `number`, `percent`, `currency`, `string`. All standard parameters can be passed via `**kwargs`.

---

## Locale System

pyneatR supports locale-aware formatting with three built-in locales: **Indian (`en_IN`)**, **US (`en_US`)**, and **EU (`de_DE`)**. Set the locale globally or pass it per-call.

### Setting the locale

```python
import pyneatR

# Set globally — all subsequent calls use this locale
pyneatR.set_locale("en_IN")

# Or pass per-call (overrides global)
pyneatR.nnumber(1_00_00_000, locale="en_IN")
```

### Indian number system (Lakhs / Crores)

When using the `en_IN` locale, pyneatR automatically applies the Indian grouping system:

```python
from pyneatR import nnumber

# Indian digit grouping: 1,00,00,000
print(nnumber(1_00_00_000, unit='', locale='en_IN', digits=0))
# "1,00,00,000"

# Indian unit labels: L (Lakh), Cr (Crore)
print(nnumber(1_50_00_000, locale='en_IN'))
# "1.5 Cr"

print(nnumber(2_50_000, locale='en_IN'))
# "2.5 L"

print(nnumber(1500, locale='en_IN'))
# "1.5 K"
```

### US locale

```python
print(nnumber(1500000, locale='en_US'))
# "1.5 M"

print(nnumber(2500000000, locale='en_US'))
# "2.5 B"
```

### EU locale (German)

```python
print(nnumber(1234.56, unit='', locale='de_DE', digits=2))
# "1.234,56"   (dot as thousand sep, comma as decimal)
```

### Timezone labels via locale

When using the smart formatter `f()` with a locale, timezone labels are automatically applied:

```python
from pyneatR import f
from datetime import datetime

ts = datetime(2025, 11, 9, 12, 7, 48)

print(f(ts, locale="en_IN"))
# "Nov 09, 2025 12H 07M 48S PM IST (Sun)"

print(f(ts, locale="en_US"))
# "Nov 09, 2025 12H 07M 48S PM EST (Sun)"

print(f(ts, locale="de_DE"))
# "Nov 09, 2025 12H 07M 48S PM CET (Sun)"
```

---

## Currency Formatting

pyneatR includes a registry of **60+ Unicode currency symbols** and the `ncurrency` function for locale-aware currency formatting.

### Basic usage

```python
from pyneatR import ncurrency

print(ncurrency(1500000, currency="USD"))
# "$1.50 Mn"

print(ncurrency(1500000, currency="EUR", locale="de_DE"))
# "1,50 Mio €"
```

### Indian Rupee with Lakhs/Crores

```python
print(ncurrency(1_50_00_000, currency="INR", locale="en_IN"))
# "₹1.50 Cr"

print(ncurrency(2_50_000, currency="INR", locale="en_IN"))
# "₹2.50 L"

# Full number without unit scaling
print(ncurrency(1_00_00_000, currency="INR", unit='', locale="en_IN", digits=0))
# "₹1,00,00,000"
```

### Looking up currency symbols

```python
from pyneatR import get_currency_symbol, CURRENCY_SYMBOLS

print(get_currency_symbol("INR"))  # ₹
print(get_currency_symbol("USD"))  # $
print(get_currency_symbol("EUR"))  # €
print(get_currency_symbol("GBP"))  # £
print(get_currency_symbol("JPY"))  # ¥
print(get_currency_symbol("BTC"))  # ₿

# Full registry available
print(len(CURRENCY_SYMBOLS))  # 60+
```

### Using via smart formatter

```python
from pyneatR import f

print(f(1_50_00_000, format_type='currency', currency="INR", locale="en_IN"))
# "₹1.50 Cr"
```

---

## DataFrame Auto-Formatting

pyneatR can automatically format Pandas and Polars DataFrames for display, inferring column types from dtypes and column names.

### Pandas

```python
import pandas as pd
import pyneatR

df = pd.DataFrame({
    "revenue": [1_50_00_000, 2_30_00_000],
    "conversion_rate": [0.15, 0.22],
    "user_id": [101, 102],
    "name": ["Alice", "Bob"],
})

# Returns a styled DataFrame (display only, data unchanged)
styled = pyneatR.neat_pandas(df, locale="en_IN", currency="INR")
styled  # In Jupyter, renders a formatted table
```

Column types are **auto-inferred** from:
- **dtype**: datetime → date/timestamp, numeric → number, string → string
- **Column name**: `revenue` → currency, `rate` → percent, `user_id` → skip
- **Value range**: all values in `[-1, 1]` → likely ratio/percent

You can also set explicit column types:

```python
styled = pyneatR.neat_pandas(df, column_types={
    "revenue": "currency",
    "conversion_rate": "percent",
}, locale="en_IN", currency="INR")
```

### Polars

```python
import polars as pl
import pyneatR

df = pl.DataFrame({
    "revenue": [15000000, 23000000],
    "conversion_rate": [0.15, 0.22],
    "name": ["Alice", "Bob"],
})

# Returns a Great Tables GT object for rich rendering
gt_table = pyneatR.neat_polars(df, locale="en_IN", currency="INR")
gt_table  # Renders in Jupyter
```

### One-liner auto-formatting with `activate()`

For the simplest experience, call `activate()` once in your notebook to automatically format all DataFrames:

```python
import pyneatR

# Activate — all DataFrames auto-format when displayed
pyneatR.activate(locale="en_IN", currency="INR")

df  # ← now renders with neat formatting in Jupyter

# Deactivate to restore original behavior
pyneatR.deactivate()
```

> **Note:** `activate()` patches `pd.DataFrame._repr_html_` and registers an IPython formatter for Polars. Use `deactivate()` to restore original behavior.

---

## API Reference

### Core Formatters

| Function | Description |
|----------|-------------|
| `nnumber(x, ...)` | Format numbers with units (K, Mn, Bn, Tn) |
| `npercent(x, ...)` | Format percentages with optional growth/bps |
| `ndate(x, ...)` | Format dates as `mmm dd, yyyy` |
| `ntimestamp(x, ...)` | Format timestamps with 12H time |
| `nday(x, ...)` | Format day names with relative context |
| `nstring(x, ...)` | Clean and format strings |
| `ncurrency(x, ...)` | Format currency with symbol and locale |
| `f(x, ...)` | Smart formatter with type inference |

### Locale Management

| Function | Description |
|----------|-------------|
| `set_locale("en_IN")` | Set global locale |
| `get_locale()` | Get current locale |
| `reset_locale()` | Reset to default (no locale) |

### DataFrame Hooks

| Function | Description |
|----------|-------------|
| `neat_pandas(df, ...)` | Returns `pd.Styler` with formatting |
| `neat_polars(df, ...)` | Returns `great_tables.GT` with formatting |
| `activate(...)` | Auto-format all DataFrames in Jupyter |
| `deactivate()` | Restore original DataFrame display |

---

## License

MIT
