#!/usr/bin/env python3
"""
Test suite for hash rate performance optimizations.

This test validates that the optimizations maintain functionality while
improving performance.
"""
import time
import numpy as np
import sys

# Import the optimized components
from zephyr_ultimate_ai_miner_fixed import (
    QuantumSimulator, NeuralNetwork, FeatureExtractor, MinerPrototype, CONFIG
)


def test_quantum_sampling():
    """Test that quantum sampling still produces valid nonces."""
    print("Testing quantum sampling optimization...")
    quantum = QuantumSimulator(qubits=32)
    
    # Test basic sampling
    nonces = quantum.sample_nonces(n=512, difficulty_hint=1.0)
    assert len(nonces) == 512, f"Expected 512 nonces, got {len(nonces)}"
    assert nonces.dtype == np.uint64, f"Expected uint64 dtype, got {nonces.dtype}"
    
    # Test with different difficulty hints
    nonces_low = quantum.sample_nonces(n=256, difficulty_hint=0.1)
    nonces_high = quantum.sample_nonces(n=256, difficulty_hint=10.0)
    assert len(nonces_low) == 256
    assert len(nonces_high) == 256
    
    print("✓ Quantum sampling works correctly")


def test_feature_extraction():
    """Test that feature extraction is vectorized and produces correct output."""
    print("\nTesting feature extraction optimization...")
    extractor = FeatureExtractor(out_dim=128)
    
    # Test single extraction
    nonce = 12345678
    timestamp = time.time()
    difficulty = 1.0
    features_single = extractor.extract(nonce, difficulty, timestamp)
    assert features_single.shape == (128,), f"Expected shape (128,), got {features_single.shape}"
    
    # Test batch extraction
    nonces = np.array([12345678, 87654321, 11111111, 99999999], dtype=np.uint64)
    features_batch = extractor.extract_batch(nonces, difficulty, timestamp)
    assert features_batch.shape == (4, 128), f"Expected shape (4, 128), got {features_batch.shape}"
    
    # Verify batch extraction matches single extraction
    features_single_from_batch = features_batch[0]
    np.testing.assert_array_almost_equal(
        features_single, features_single_from_batch, decimal=5,
        err_msg="Batch extraction doesn't match single extraction"
    )
    
    print("✓ Feature extraction works correctly (single and batch)")


def test_neural_network_selective_hashing():
    """Test that neural network with selective hashing produces valid predictions."""
    print("\nTesting neural network with selective hashing...")
    layers = [128, 256, 128, 64, 32, 16, 8, 1]
    nn = NeuralNetwork(layers=layers, lr=0.001)
    
    # Test single sample prediction
    x_single = np.random.random(128).astype(np.float32)
    pred_single = nn.predict(x_single)
    assert pred_single.shape == (1,), f"Expected shape (1,), got {pred_single.shape}"
    assert 0 <= pred_single[0] <= 1, f"Prediction out of bounds: {pred_single[0]}"
    
    # Test batch prediction
    x_batch = np.random.random((16, 128)).astype(np.float32)
    pred_batch = nn.predict(x_batch)
    assert pred_batch.shape == (16,), f"Expected shape (16,), got {pred_batch.shape}"
    assert all(0 <= p <= 1 for p in pred_batch), "Some predictions out of bounds"
    
    # Verify volcano blast history is being recorded
    assert len(nn.volcano_blast_history) > 0, "Volcano blast history not recorded"
    
    print("✓ Neural network with selective hashing works correctly")


def test_hash_caching():
    """Test that hash result caching works correctly."""
    print("\nTesting hash result caching...")
    config = CONFIG.copy()
    config["ENABLE_QUANTUM_AI"] = True
    config["ENABLE_NEURAL_NETWORKS"] = True
    
    miner = MinerPrototype(config)
    
    # Test cache storage and retrieval
    nonce = 12345678
    timestamp = time.time()
    difficulty = 1.0
    
    # First access should return None (cache miss)
    cached = miner._get_cached_hash(nonce, timestamp, difficulty)
    assert cached is None, "Cache should be empty initially"
    
    # Store a result
    test_result = (b"test_hash", {"metadata": "test"})
    miner._cache_hash(nonce, timestamp, difficulty, test_result)
    
    # Second access should return the cached value
    cached = miner._get_cached_hash(nonce, timestamp, difficulty)
    assert cached is not None, "Cache should return stored value"
    assert cached[0] == b"test_hash", "Cached value doesn't match"
    
    # Test LRU eviction (cache has max size of 1000)
    for i in range(1100):
        miner._cache_hash(i, timestamp, difficulty, (b"hash", {}))
    
    assert len(miner.hash_cache) <= 1000, f"Cache size exceeded limit: {len(miner.hash_cache)}"
    
    print("✓ Hash caching works correctly with LRU eviction")


def test_async_api_threads():
    """Test that async API queries don't block the main thread."""
    print("\nTesting async API query mechanism...")
    config = CONFIG.copy()
    config["ENABLE_XMRIG_API"] = True
    config["XMRIG_API_URL"] = "http://127.0.0.1:18081"  # Will fail, but that's ok for testing
    
    miner = MinerPrototype(config)
    
    # Start async query
    start_time = time.time()
    miner._start_api_query()
    elapsed = time.time() - start_time
    
    # Should return immediately (not block)
    assert elapsed < 0.1, f"Async query blocked for {elapsed:.3f}s"
    
    # Wait for thread to complete
    if miner.api_thread:
        miner.api_thread.join(timeout=2.0)
    
    print("✓ Async API queries are non-blocking")


def test_full_mining_cycle():
    """Test a complete mining cycle with all optimizations."""
    print("\nTesting full mining cycle with optimizations...")
    config = CONFIG.copy()
    config["ENABLE_QUANTUM_AI"] = True
    config["ENABLE_NEURAL_NETWORKS"] = True
    config["NEURAL_LAYERS"] = [128, 256, 128, 64, 32, 16, 8, 1]
    
    miner = MinerPrototype(config)
    
    # Run a single cycle
    start_time = time.time()
    miner.run_cycle(simulate=True)
    elapsed = time.time() - start_time
    
    print(f"  Mining cycle completed in {elapsed:.3f}s")
    
    # Verify neural network state
    assert len(miner.nn.volcano_blast_history) > 0, "Volcano blast not executed"
    assert miner.nn.waterfall_cascade_state is not None, "Waterfall cascade not executed"
    
    # Verify samples collected
    assert len(miner.samples_buffer) > 0, "No samples collected"
    
    print("✓ Full mining cycle works correctly with all optimizations")


def performance_comparison():
    """Compare performance of batch vs sequential feature extraction."""
    print("\n" + "="*60)
    print("Performance Comparison: Batch vs Sequential Feature Extraction")
    print("="*60)
    
    extractor = FeatureExtractor(out_dim=128)
    timestamp = time.time()
    difficulty = 1.0
    nonces = np.random.randint(0, 2**32, size=512, dtype=np.uint64)
    
    # Time sequential extraction (simulating old code)
    start = time.time()
    features_seq = []
    for nonce in nonces:
        features_seq.append(extractor.extract(int(nonce), difficulty, timestamp))
    features_seq = np.stack(features_seq)
    seq_time = time.time() - start
    
    # Time batch extraction (optimized code)
    start = time.time()
    features_batch = extractor.extract_batch(nonces, difficulty, timestamp)
    batch_time = time.time() - start
    
    speedup = seq_time / batch_time
    print(f"Sequential extraction: {seq_time:.4f}s")
    print(f"Batch extraction:      {batch_time:.4f}s")
    print(f"Speedup:               {speedup:.2f}x")
    
    # Verify results are the same
    np.testing.assert_array_almost_equal(
        features_seq, features_batch, decimal=5,
        err_msg="Batch and sequential results don't match"
    )
    
    print("✓ Batch extraction is faster and produces identical results")


def main():
    """Run all tests."""
    print("="*60)
    print("TRUEQUANTUM Hash Rate Optimization Test Suite")
    print("="*60)
    
    try:
        test_quantum_sampling()
        test_feature_extraction()
        test_neural_network_selective_hashing()
        test_hash_caching()
        test_async_api_threads()
        test_full_mining_cycle()
        performance_comparison()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
