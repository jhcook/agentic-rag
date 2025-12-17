#!/usr/bin/env python3
"""
Benchmark Concurrency: Threading vs Multiprocessing
"""

import time
import threading
import multiprocessing
import sys
import platform
import math

def cpu_bound_task(n):
    """A CPU-intensive task: heavy math calculations."""
    while n > 0:
        math.factorial(n)
        n -= 1

def run_serial(iterations, complexity):
    start = time.time()
    for _ in range(iterations):
        cpu_bound_task(complexity)
    return time.time() - start

def run_threaded(iterations, complexity):
    threads = []
    start = time.time()
    for _ in range(iterations):
        t = threading.Thread(target=cpu_bound_task, args=(complexity,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    return time.time() - start

def run_multiprocess(iterations, complexity):
    processes = []
    start = time.time()
    for _ in range(iterations):
        p = multiprocessing.Process(target=cpu_bound_task, args=(complexity,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    return time.time() - start

def main():
    print(f"Benchmark Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    
    try:
        # sys._is_gil_enabled() is new in 3.13+ for free-threaded builds
        gil_status = sys._is_gil_enabled()
        print(f"  GIL Enabled: {gil_status}")
    except AttributeError:
        print("  GIL Status: Active (Legacy build or < 3.13)")
        
    iterations = 8
    complexity = 5000  # Adjust based on CPU speed
    
    print(f"\nRunning Benchmark ({iterations} tasks)...")
    
    # Serial
    t_serial = run_serial(iterations, complexity)
    print(f"  Serial:         {t_serial:.4f}s")
    
    # Threading
    t_threads = run_threaded(iterations, complexity)
    print(f"  Threading:      {t_threads:.4f}s (Speedup vs Serial: {t_serial/t_threads:.2f}x)")
    
    # Multiprocessing
    t_procs = run_multiprocess(iterations, complexity)
    print(f"  Multiprocessing: {t_procs:.4f}s (Speedup vs Serial: {t_serial/t_procs:.2f}x)")

    print("\nInterpretation:")
    print("  - If Threading ~= Serial (Speedup ~1.0x), GIL is holding you back.")
    print("  - If Threading ~= Multiprocessing (Speedup > 1.0x), GIL is disabled or released effectively.")

if __name__ == "__main__":
    main()
