/**
 * MNIST Neural Network - GPU-Accelerated Training Script
 *
 * Architecture: 784 -> 16 -> 16 -> 10
 *
 * Features:
 *   ✓ GPU acceleration using GPU.js
 *   ✓ Input normalization (z-score)
 *   ✓ Performance metrics (images/sec, batch time)
 *
 * TODO:
 *   - Bias terms
 *   - Batch normalization
 *   - Dropout regularization
 *   - Adam optimizer (currently using basic SGD)
 *   - Learning rate scheduling
 *   - Data augmentation
 */

import fs from 'fs';
import { GPU } from 'gpu.js';
import { readMNIST } from './images.js';
import {
    relu,
    softmax,
    crossEntropyLoss,
    initializeWeights,
} from './utils.js';

// Initialize GPU.js
const gpu = new GPU();

// Hyperparameters
const learningRate = 0.01;

// Normalization statistics (calculated from MNIST training set)
const PIXEL_MEAN = 0.1307;
const PIXEL_STD = 0.3081;

/**
 * Normalize pixel values using z-score normalization
 */
function normalizePixels(pixels) {
    return pixels.map(pixel => (pixel - PIXEL_MEAN) / PIXEL_STD);
}

// Pre-compile GPU kernels for each layer size
// This avoids the overhead of creating kernels on every forward/backward pass

// Layer 0: 784 -> 16
const gpuForward0 = gpu.createKernel(function(w, inp) {
    let sum = 0;
    for (let j = 0; j < 784; j++) {
        sum += w[this.thread.x][j] * inp[j];
    }
    return sum;
}).setOutput([16]);

const gpuBackward0 = gpu.createKernel(function(w, d) {
    let sum = 0;
    for (let j = 0; j < 16; j++) {
        sum += w[j][this.thread.x] * d[j];
    }
    return sum;
}).setOutput([784]);

// Layer 1: 16 -> 16
const gpuForward1 = gpu.createKernel(function(w, inp) {
    let sum = 0;
    for (let j = 0; j < 16; j++) {
        sum += w[this.thread.x][j] * inp[j];
    }
    return sum;
}).setOutput([16]);

const gpuBackward1 = gpu.createKernel(function(w, d) {
    let sum = 0;
    for (let j = 0; j < 16; j++) {
        sum += w[j][this.thread.x] * d[j];
    }
    return sum;
}).setOutput([16]);

// Layer 2: 16 -> 10
const gpuForward2 = gpu.createKernel(function(w, inp) {
    let sum = 0;
    for (let j = 0; j < 16; j++) {
        sum += w[this.thread.x][j] * inp[j];
    }
    return sum;
}).setOutput([10]);

const gpuBackward2 = gpu.createKernel(function(w, d) {
    let sum = 0;
    for (let j = 0; j < 10; j++) {
        sum += w[j][this.thread.x] * d[j];
    }
    return sum;
}).setOutput([16]);

/**
 * Forward pass through a layer - optimized with pre-compiled GPU kernels
 */
function forwardLayerGPU(input, weights, layer) {
    let result;
    if (layer === 0) {
        result = gpuForward0(weights, input);
    } else if (layer === 1) {
        result = gpuForward1(weights, input);
    } else {
        result = gpuForward2(weights, input);
    }
    return Array.from(result);
}

/**
 * Apply ReLU activation - keeping on CPU as it's element-wise and very fast
 */
function applyReLUGPU(input) {
    return input.map(x => x > 0 ? x : 0);
}

/**
 * Backpropagate error through a layer
 */
function backpropErrorGPU(weights, delta, layer) {
    let result;
    if (layer === 2) {
        result = gpuBackward2(weights, delta);
    } else if (layer === 1) {
        result = gpuBackward1(weights, delta);
    } else {
        result = gpuBackward0(weights, delta);
    }
    return Array.from(result);
}

/**
 * Apply ReLU derivative - keeping on CPU
 */
function applyReLUDerivativeGPU(gradient, z) {
    return gradient.map((g, i) => z[i] > 0 ? g : 0);
}

function loadWeights() {
    let w = {};
    let exists = fs.existsSync('./w_gpu.json');
    if (exists) {
        w = fs.readFileSync('./w_gpu.json');
        w = JSON.parse(w);
    } else {
        w = {
            0: initializeWeights(784, 16),
            1: initializeWeights(16, 16),
            2: initializeWeights(16, 10),
        }
    }
    return w;
}

function createZeroMatrix(rows, cols) {
    return Array(rows).fill(0).map(() => Array(cols).fill(0));
}

function run(w, position) {
    // Load batch of images
    let images = readMNIST(position*10, position*10+10);

    // Accumulated weight gradients for the entire batch
    let weightGradients = {
        0: createZeroMatrix(16, 784),
        1: createZeroMatrix(16, 16),
        2: createZeroMatrix(10, 16)
    };

    let totalLoss = 0;
    let correctPredictions = 0;

    // Process each image in the batch
    images.forEach(function (image) {
        // Create one-hot encoded target
        let target = new Array(10).fill(0);
        target[image.label] = 1;

        // ========== NORMALIZATION ==========
        const normalizedPixels = normalizePixels(image.pixels);

        // ========== FORWARD PASS - GPU ACCELERATED ==========

        // Layer 0: Input pixels (784) -> Hidden layer 1 (16 neurons) with ReLU
        let z0 = forwardLayerGPU(normalizedPixels, w[0], 0);
        let a0 = applyReLUGPU(z0);

        // Layer 1: Hidden layer 1 (16) -> Hidden layer 2 (16 neurons) with ReLU
        let z1 = forwardLayerGPU(a0, w[1], 1);
        let a1 = applyReLUGPU(z1);

        // Layer 2: Hidden layer 2 (16) -> Output layer (10 neurons)
        let z2 = forwardLayerGPU(a1, w[2], 2);
        let output = softmax(z2);

        // Calculate loss
        totalLoss += crossEntropyLoss(output, target);

        // Check if prediction is correct
        const predictedLabel = output.indexOf(Math.max(...output));
        if (predictedLabel === image.label) {
            correctPredictions++;
        }

        // ========== BACKWARD PASS - GPU ACCELERATED ==========

        // Output layer gradient
        let dz2 = output.map((prob, i) => prob - target[i]);

        // Accumulate weight gradients for layer 2
        for (let i = 0; i < w[2].length; i++) {
            for (let j = 0; j < w[2][i].length; j++) {
                weightGradients[2][i][j] += dz2[i] * a1[j];
            }
        }

        // Backpropagate to hidden layer 2
        let da1 = backpropErrorGPU(w[2], dz2, 2);

        // Apply ReLU derivative
        let dz1 = applyReLUDerivativeGPU(da1, z1);

        // Accumulate weight gradients for layer 1
        for (let i = 0; i < w[1].length; i++) {
            for (let j = 0; j < w[1][i].length; j++) {
                weightGradients[1][i][j] += dz1[i] * a0[j];
            }
        }

        // Backpropagate to hidden layer 1
        let da0 = backpropErrorGPU(w[1], dz1, 1);

        // Apply ReLU derivative
        let dz0 = applyReLUDerivativeGPU(da0, z0);

        // Accumulate weight gradients for layer 0
        for (let i = 0; i < w[0].length; i++) {
            for (let j = 0; j < w[0][i].length; j++) {
                weightGradients[0][i][j] += dz0[i] * normalizedPixels[j];
            }
        }
    });

    // ========== UPDATE WEIGHTS - CPU (could be GPU optimized further) ==========
    const batchSize = images.length;

    for (let layer = 0; layer <= 2; layer++) {
        for (let i = 0; i < w[layer].length; i++) {
            for (let j = 0; j < w[layer][i].length; j++) {
                const gradient = weightGradients[layer][i][j] / batchSize;
                w[layer][i][j] -= learningRate * gradient;
            }
        }
    }

    // Calculate accuracy for this batch
    const accuracy = (correctPredictions / batchSize) * 100;

    return { loss: totalLoss / batchSize, accuracy };
}

function main() {
    // Load weights
    let w = loadWeights();

    const totalBatches = 1000;
    const reportInterval = 10;
    let recentMetrics = [];

    // Performance tracking
    let totalImagesProcessed = 0;
    let totalTimeMs = 0;
    const startTime = Date.now();

    console.log('=' .repeat(70));
    console.log('GPU-ACCELERATED MNIST TRAINING');
    console.log('=' .repeat(70));
    console.log(`Total batches: ${totalBatches}`);
    console.log(`Batch size: 10 images`);
    console.log(`Total images: ${totalBatches * 10}`);
    console.log('=' .repeat(70));
    console.log('');

    for (let i = 0; i < totalBatches; i++) {
        const batchStartTime = Date.now();
        const metrics = run(w, i);
        const batchEndTime = Date.now();

        const batchTimeMs = batchEndTime - batchStartTime;
        const imagesPerSecond = (10 / batchTimeMs) * 1000;

        totalImagesProcessed += 10;
        totalTimeMs += batchTimeMs;

        recentMetrics.push({
            ...metrics,
            batchTimeMs,
            imagesPerSecond
        });

        // Print batch results with performance metrics
        console.log(
            `Batch ${String(i).padStart(4, ' ')} | ` +
            `Loss: ${metrics.loss.toFixed(4)} | ` +
            `Acc: ${metrics.accuracy.toFixed(1)}% | ` +
            `Time: ${batchTimeMs.toFixed(0)}ms | ` +
            `Speed: ${imagesPerSecond.toFixed(1)} img/s`
        );

        // Print summary every reportInterval batches
        if ((i + 1) % reportInterval === 0) {
            const avgLoss = recentMetrics.reduce((a, b) => a + b.loss, 0) / recentMetrics.length;
            const avgAccuracy = recentMetrics.reduce((a, b) => a + b.accuracy, 0) / recentMetrics.length;
            const avgBatchTime = recentMetrics.reduce((a, b) => a + b.batchTimeMs, 0) / recentMetrics.length;
            const avgSpeed = recentMetrics.reduce((a, b) => a + b.imagesPerSecond, 0) / recentMetrics.length;
            const progress = ((i + 1) / totalBatches * 100).toFixed(1);

            const elapsedTime = Date.now() - startTime;
            const overallSpeed = (totalImagesProcessed / elapsedTime) * 1000;

            console.log('');
            console.log('=' .repeat(70));
            console.log(`SUMMARY - Batches ${i - reportInterval + 2}-${i + 1} (Progress: ${progress}%)`);
            console.log('-' .repeat(70));
            console.log(`Avg Loss: ${avgLoss.toFixed(4)} | Avg Accuracy: ${avgAccuracy.toFixed(1)}%`);
            console.log(`Avg Batch Time: ${avgBatchTime.toFixed(0)}ms | Avg Speed: ${avgSpeed.toFixed(1)} img/s`);
            console.log(`Overall Speed: ${overallSpeed.toFixed(1)} img/s | Total Time: ${(elapsedTime / 1000).toFixed(1)}s`);
            console.log('=' .repeat(70));
            console.log('');

            recentMetrics = [];
        }
    }

    const totalTime = Date.now() - startTime;
    const overallSpeed = (totalImagesProcessed / totalTime) * 1000;

    console.log('');
    console.log('=' .repeat(70));
    console.log('TRAINING COMPLETE');
    console.log('=' .repeat(70));
    console.log(`Total images processed: ${totalImagesProcessed}`);
    console.log(`Total time: ${(totalTime / 1000).toFixed(2)}s`);
    console.log(`Average speed: ${overallSpeed.toFixed(1)} images/second`);
    console.log(`Average batch time: ${(totalTime / totalBatches).toFixed(0)}ms`);
    console.log('=' .repeat(70));

    console.log('\nSaving weights to w_gpu.json...');
    fs.writeFileSync('./w_gpu.json', JSON.stringify(w, null, 4));
    console.log('Weights saved successfully!');
    console.log('\nRun "node test.js" to test the trained network.');
}

main();
