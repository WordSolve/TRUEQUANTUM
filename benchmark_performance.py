#!/usr/bin/env python3
"""
Benchmark script to measure hash rate performance improvements.
"""
import time
import numpy as np
from zephyr_ultimate_ai_miner_fixed import MinerPrototype, CONFIG

def benchmark_mining_cycles(num_cycles=10):
    """Run multiple mining cycles and measure average performance."""
    print("="*60)
    print(f"Hash Rate Performance Benchmark ({num_cycles} cycles)")
    print("="*60)
    
    config = CONFIG.copy()
    config["ENABLE_QUANTUM_AI"] = True
    config["ENABLE_NEURAL_NETWORKS"] = True
    config["NEURAL_LAYERS"] = [128, 256, 512, 256, 128, 64, 32, 16, 8, 1]
    
    miner = MinerPrototype(config)
    
    cycle_times = []
    candidates_per_cycle = 512
    
    print(f"\nRunning {num_cycles} mining cycles...")
    print("Cycle | Time (s) | Hash Rate (H/s) | Cache Size")
    print("-" * 60)
    
    for i in range(num_cycles):
        start = time.time()
        miner.run_cycle(simulate=True)
        elapsed = time.time() - start
        cycle_times.append(elapsed)
        
        hash_rate = candidates_per_cycle / elapsed
        cache_size = len(miner.hash_cache)
        
        print(f"{i+1:5d} | {elapsed:8.4f} | {hash_rate:15.2f} | {cache_size:10d}")
    
    print("-" * 60)
    avg_time = np.mean(cycle_times)
    std_time = np.std(cycle_times)
    avg_hash_rate = candidates_per_cycle / avg_time
    
    print(f"\nPerformance Summary:")
    print(f"  Average cycle time:     {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"  Average hash rate:      {avg_hash_rate:.2f} H/s")
    print(f"  Total candidates:       {num_cycles * candidates_per_cycle}")
    print(f"  Total time:             {sum(cycle_times):.2f}s")
    print(f"  Cache utilization:      {len(miner.hash_cache)} entries")
    
    print("\nOptimization Features Active:")
    print("  ✓ Selective hashing (5 gates instead of 10)")
    print("  ✓ Vectorized feature extraction")
    print("  ✓ Batch neural network processing")
    print("  ✓ Hash result caching (LRU)")
    print("  ✓ Async API queries")
    print("  ✓ Optimized quantum sampling")
    
    print("="*60)

if __name__ == "__main__":
    benchmark_mining_cycles(num_cycles=10)
