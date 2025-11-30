# Performance Comparison: CPU vs GPU Training

This document compares the performance of CPU-based and GPU-accelerated training for the MNIST neural network.

## Files

- **index.js** - CPU-based training (original)
- **gpu.js** - GPU-accelerated training using GPU.js

Both scripts save weights to separate files:
- CPU version: `w.json`
- GPU version: `w_gpu.json`

## How to Run

```bash
# CPU training
node index.js

# GPU training
node gpu.js
```

## Performance Metrics

Both scripts display real-time performance metrics:

- **Batch time** - Time taken to process one batch (10 images)
- **Speed** - Images processed per second (img/s)
- **Loss** - Training loss value
- **Accuracy** - Batch accuracy percentage
- **Overall Speed** - Average speed across all processed batches
- **Total Time** - Elapsed time since training started

### Summary Display

Every 10 batches, you'll see a summary like:

```
======================================================================
SUMMARY - Batches 11-20 (Progress: 2.0%)
----------------------------------------------------------------------
Avg Loss: 2.0999 | Avg Accuracy: 19.0%
Avg Batch Time: 25ms | Avg Speed: 395.3 img/s
Overall Speed: 178.4 img/s | Total Time: 1.1s
======================================================================
```

## GPU Optimization Details

The GPU version uses GPU.js to accelerate:

1. **Matrix-vector multiplications** - Forward and backward passes through layers
2. **Pre-compiled kernels** - Kernels are created once at startup, not per-operation
3. **Layer-specific kernels** - Each layer has its own optimized kernel:
   - Layer 0: 784 → 16 neurons
   - Layer 1: 16 → 16 neurons
   - Layer 2: 16 → 10 neurons

### What Runs on GPU vs CPU

**GPU:**
- Forward pass matrix multiplications
- Backpropagation matrix multiplications

**CPU (kept on CPU for performance):**
- ReLU activation functions
- ReLU derivatives
- Softmax
- Loss calculation
- Weight updates
- Element-wise operations

Small element-wise operations are faster on CPU due to GPU transfer overhead.

## Expected Performance

Typical performance you can expect:

**GPU Version (gpu.js):**
- ~280-400 images/second (after initial warmup)
- First batch slower (~500ms) due to GPU kernel compilation
- Subsequent batches: ~25-40ms each

**CPU Version (index.js):**
- Performance varies by CPU
- Typical range: 50-150 images/second on modern CPUs

### Performance Factors

GPU performance depends on:
- GPU model and memory
- GPU.js backend (WebGL vs headless-gl)
- Batch size (currently 10 images)
- Network architecture size

## Comparing Results

To compare training effectiveness:

1. Run both scripts for the same number of batches
2. Compare final loss and accuracy values
3. Both should converge to similar results (they use identical algorithms)
4. GPU version should be significantly faster

## Notes

- First batch is always slower due to initialization
- GPU.js uses WebGL or headless-gl depending on environment
- Performance improves after warmup period (first 10-20 batches)
- Larger batch sizes may improve GPU utilization but require more memory
