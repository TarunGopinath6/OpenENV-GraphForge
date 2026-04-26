import time
import numpy as np
import string
import random
from pyneatR import nnumber, npercent, ndate, ntimestamp, nstring, nday

def benchmark(name, func, data, **kwargs):
    start = time.time()
    res = func(data, **kwargs)
    end = time.time()
    print(f"{name}: {end - start:.4f} seconds for {len(data)} items.")
    return res

def run_benchmarks():
    N = 1_000_000
    print(f"Generating {N} random items for each test...")

    # Numbers
    numbers = np.random.uniform(0, 1e9, N)
    benchmark("nnumber (default)", nnumber, numbers)
    benchmark("nnumber (auto unit)", nnumber, numbers, unit='auto')

    # Percents
    percents = np.random.uniform(-5, 5, N) # -500% to +500%
    benchmark("npercent", npercent, percents)

    # Dates
    # Random days from 1970 to 2050
    days = np.random.randint(0, 30000, N)
    dates = np.array(days, dtype='datetime64[D]')
    benchmark("ndate", ndate, dates)
    
    # Timestamps
    # Random seconds
    seconds = np.random.randint(0, 2000000000, N)
    timestamps = np.array(seconds, dtype='datetime64[s]')
    benchmark("ntimestamp", ntimestamp, timestamps)
    
    # Strings
    # Generate random strings? Too slow to generate in python loop for 1M.
    # Use numpy char ?
    # Let's create a small set of unique strings and repeat them, to simulate real data
    # Real data often has low cardinality for categorical strings.
    # The unique_optimization shines there.
    # If standard unique_optimization is assumed, we should test with some dups.
    
    unique_count = 10000
    # Create 10k random strings
    params = list(string.ascii_letters + " !@#")
    # This generation is slow in python, so let's keep it simple.
    chars = np.array(list(string.ascii_letters), dtype='U1')
    # Generate 10k words of length 10
    # vectorized random choice
    # indices = np.random.randint(0, len(chars), (unique_count, 10))
    # words = "".join... too complex.
    
    # Simple list comp
    vocab = ["".join(random.choices(string.ascii_letters + "  !@#", k=10)) for _ in range(unique_count)]
    
    # Now sample N from vocab
    strings_data = np.random.choice(vocab, N)
    
    benchmark("nstring (full)", nstring, strings_data, case='title', remove_specials=True)
    
    # Test with High Cardinality (Worst case for optimization?)
    # If 1M unique strings, unique_optimization adds overhead of finding unique + sorting.
    # And then processing 1M items.
    # But string processing is expensive anyway.
    
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_benchmarks()
