import fs from 'fs';
import {
    crossEntropyLoss,
    relu,
    softmax
} from './utils.js';
import {readMNIST} from "./images.js";

/**
 * Initialize weights using He initialization (Kaiming initialization)
 *
 * He initialization is specifically designed for layers with ReLU activation.
 * It helps prevent vanishing/exploding gradients by scaling weights appropriately.
 *
 * Formula: w ~ N(0, sqrt(2/n_in))
 * Where:
 *   - w = weight value
 *   - N(μ, σ) = Normal distribution with mean μ and std deviation σ
 *   - n_in = number of input connections to the neuron
 *
 * The factor of 2 in the numerator accounts for ReLU's property of zeroing out
 * half the neurons on average, which would otherwise halve the variance.
 *
 * @param {number} numInputs - Number of inputs to the neuron
 * @returns {Array<number>} Array of initialized weights
 */
const initializeWeights = (numInputs) => {
    // Standard deviation for He initialization
    // σ = sqrt(2 / n_in)
    const stdDev = Math.sqrt(2 / numInputs);
    const weights = [];

    for (let i = 0; i < numInputs; i++) {
        // Box-Muller transform to generate Gaussian random numbers
        // Converts uniform random numbers to normally distributed values
        // Formula: z = sqrt(-2*ln(u1)) * cos(2*π*u2)
        // Where u1, u2 ~ U(0,1) (uniform distribution)
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);

        // Scale by standard deviation to get N(0, σ)
        weights[i] = z * stdDev;
    }

    return weights;
}

/**
 * Create a single neuron in a neural network layer
 *
 * A neuron performs a weighted sum of its inputs followed by an activation function.
 * This is the fundamental building block of a neural network.
 *
 * Forward propagation formula:
 *   z = Σ(w_i * x_i) for i=0 to n-1
 *   a = f(z)
 *
 * Where:
 *   - z = pre-activation value (raw weighted sum)
 *   - w_i = weight for input i
 *   - x_i = input value i
 *   - a = activation value (output after applying activation function)
 *   - f = activation function (e.g., ReLU, sigmoid)
 *   - n = number of inputs
 *
 * @param {Object} previousLayer - The layer that feeds into this neuron
 * @returns {Object} Neuron object with calculate method
 */
const Neuron = function(previousLayer) {
    let size = previousLayer.value.length;
    let input = previousLayer.value; // Reference to previous layer's output

    return {
        value: null,     // Post-activation value: a = f(z)
        raw: null,       // Pre-activation value: z = Σ(w_i * x_i)
        input: input,    // Reference to input values
        weights: initializeWeights(size), // Weight vector [w_0, w_1, ..., w_n]

        /**
         * Calculate neuron output via forward propagation
         *
         * Step 1: Compute weighted sum (dot product)
         *   z = w·x = Σ(w_i * x_i)
         *
         * Step 2: Apply activation function
         *   a = f(z)
         *
         * @param {Function} activation - Activation function to apply (optional)
         * @returns {number} The neuron's output value
         */
        calculate: function(activation) {
            // Compute dot product: z = w·x
            // Inlined for performance (avoids function call overhead)
            let sum = 0;
            for (let i = 0; i < size; i++) {
                sum += input[i] * this.weights[i];
            }
            this.raw = sum; // Store pre-activation for backprop

            // Apply activation function if provided: a = f(z)
            // Otherwise use identity function: a = z
            return this.value = activation ? activation(sum) : sum;
        }
    }
}

const Layer = function(numOfNeurons, previousLayer, activation) {
    const neurons = [];

    // Neurons linked to the previous layers. First layer has no neurons
    if (previousLayer) {
        for (let i = 0; i < numOfNeurons; i++) {
            neurons.push(Neuron(previousLayer))
        }
    }

    // weightGradient[neuronIndex][inputIndex]
    // Shape: [numOfNeurons][numInputsPerNeuron]
    const weightGradient = [];


    const value = new Array(numOfNeurons).fill(0);
    const error = new Array(numOfNeurons).fill(0);

    const layer = {
        value: value,
        error: error,
        neurons: neurons,
        activation: activation,
        calculate: function() {
            // Calculate all neurons
            for (let j = 0; j < neurons.length; j++) {
                this.value[j] = neurons[j].calculate(activation);
            }
        },
        /**
         * Update weights using gradient descent
         *
         * Gradient descent adjusts weights to minimize the loss function.
         * For mini-batch gradient descent, we average gradients across the batch.
         *
         * Update rule:
         *   w_new = w_old - α * (1/m) * Σ(∇w)
         *
         * Where:
         *   - w = weight value
         *   - α = learning rate (step size)
         *   - m = batch size (number of examples in mini-batch)
         *   - Σ(∇w) = sum of gradients across batch (weightGradient)
         *   - (1/m) * Σ(∇w) = average gradient
         *
         * The negative sign moves weights in the direction opposite to the gradient,
         * which is the direction of steepest descent (reduces loss).
         *
         * @param {number} learningRate - Step size for gradient descent (α)
         * @param {number} batchSize - Number of examples in the mini-batch (m)
         */
        updateWeights: function(learningRate, batchSize) {
            // Pre-calculate 1/batchSize to avoid division in inner loop
            // Convert division to multiplication (faster operation)
            const invBatchSize = 1 / batchSize;

            for (let i = 0; i < numOfNeurons; i++) {
                let weights = neurons[i].weights;
                for (let j = 0; j < weights.length; j++) {
                    // Gradient descent update: w = w - α * (1/m) * Σ(∇w)
                    weights[j] -= learningRate * (weightGradient[i][j] * invBatchSize);
                }
            }
        }
    }


    if (previousLayer) {
        const e = previousLayer.error;
        const v = previousLayer.value;
        const n = previousLayer.neurons;
        const numOfInputs = v.length;
        for (let i = 0; i < numOfNeurons; i++) {
            weightGradient[i] = new Array(numOfInputs).fill(0);
        }

        /**
         * Compute gradients via backpropagation
         *
         * Backpropagation computes how the loss changes with respect to each weight
         * by applying the chain rule from calculus.
         *
         * For a neuron in layer L:
         *   δ^L = error signal for layer L (gradient of loss w.r.t. pre-activation)
         *   a^(L-1) = activation from previous layer
         *   w^L = weights of current layer
         *
         * STEP 1: Compute weight gradients for THIS layer
         *   ∂Loss/∂w_ij = δ^L_i * a^(L-1)_j
         *
         *   Where:
         *     - i = index of neuron in current layer
         *     - j = index of input from previous layer
         *     - δ^L_i = error signal for neuron i (this.error[i])
         *     - a^(L-1)_j = activation from previous layer (v[j])
         *
         * STEP 2: Backpropagate error to previous layer
         *   δ^(L-1)_j = Σ(w_ij * δ^L_i) for all i in current layer
         *
         *   This computes how much each neuron in the previous layer
         *   contributed to the error (before applying activation derivative).
         *
         * STEP 3: Apply activation function derivative
         *   For ReLU: f'(z) = 1 if z > 0, else 0
         *
         *   Final error: δ^(L-1)_j = δ^(L-1)_j * f'(z^(L-1)_j)
         *
         *   Note: We apply this AFTER accumulating all errors to handle
         *   the case where multiple neurons feed into the same previous neuron.
         */
        layer.gradient = function() {
            // Reset previous layer errors to zero before accumulation
            for (let j = 0; j < e.length; j++) {
                e[j] = 0;
            }

            // Backpropagation: compute gradients and propagate errors
            for (let i = 0; i < numOfNeurons; i++) {
                for (let j = 0; j < numOfInputs; j++) {
                    // STEP 1: Weight gradient for THIS layer
                    // ∂Loss/∂w_ij = δ^L_i * a^(L-1)_j
                    weightGradient[i][j] += this.error[i] * v[j];

                    // STEP 2: Backpropagate error to previous layer
                    // δ^(L-1)_j += w_ij * δ^L_i
                    e[j] += neurons[i].weights[j] * this.error[i];
                }
            }

            // STEP 3: Apply ReLU derivative (only for hidden layers with neurons)
            // ReLU'(z) = 1 if z > 0, else 0
            // If z ≤ 0, set error to 0 (gradient doesn't flow through dead neurons)
            if (n) {
                for (let j = 0; j < n.length; j++) {
                    if (n[j].raw <= 0) {
                        e[j] = 0; // Zero gradient when ReLU is inactive
                    }
                }
            }
        }

        /**
         * Reset accumulated gradients to zero
         *
         * In mini-batch gradient descent, we accumulate gradients across
         * multiple training examples before updating weights. After updating,
         * we must reset gradients to zero for the next batch.
         *
         * This should be called at the start of each mini-batch.
         */
        layer.resetGradient = function() {
            for (let i = 0; i < numOfNeurons; i++) {
                for (let j = 0; j < numOfInputs; j++) {
                    weightGradient[i][j] = 0;
                }
            }
        }
    }

    return layer;
}

/**
 * Create a feedforward neural network
 *
 * A feedforward neural network consists of layers of neurons where information
 * flows from input to output without cycles. Training uses backpropagation
 * and gradient descent to adjust weights.
 *
 * Network architecture:
 *   Input Layer → Hidden Layer(s) → Output Layer
 *
 * @param {Array<number>} size - Array defining network architecture
 *                               e.g., [784, 128, 10] = 784 inputs, 128 hidden, 10 outputs
 * @param {Function} activationMethod - Activation function for hidden layers (e.g., ReLU)
 * @returns {Object} Network object with training and inference methods
 */
const Network = function(size, activationMethod) {
    const layers = [];

    // Create all layers
    // Hidden layers use the specified activation function
    // Output layer uses no activation (raw logits for softmax)
    for (let i = 0; i < size.length; i++) {
        // All layers except output use activation function
        let activation = i < size.length - 1 ? activationMethod : null;
        layers.push(Layer(size[i], layers[i-1], activation))
    }

    return {
        layers: layers,

        /**
         * Get network output, optionally applying a transformation
         *
         * @param {Function} convert - Optional function to apply (e.g., softmax)
         * @returns {Array<number>} Network output values
         */
        output: function(convert) {
            let output = layers[size.length-1].value;
            if (convert) {
                // Apply transformation (e.g., softmax for probabilities)
                output = convert(output);
            }
            return output;
        },

        /**
         * Forward propagation: compute network output from input
         *
         * Passes input through each layer sequentially:
         *   Layer 0 (input) → Layer 1 (hidden) → ... → Layer L (output)
         *
         * Each layer computes:
         *   z^l = W^l · a^(l-1) + b^l
         *   a^l = f(z^l)
         *
         * Where:
         *   - z^l = pre-activation (weighted sum)
         *   - W^l = weight matrix for layer l
         *   - a^(l-1) = activations from previous layer
         *   - b^l = bias (implicit, set to 0 in this implementation)
         *   - f = activation function
         *   - a^l = post-activation output
         *
         * @param {Array<number>} input - Input vector (e.g., flattened image)
         */
        forward: function(input) {
            // Set input layer values
            for (let i = 0; i < input.length; i++) {
                layers[0].value[i] = input[i];
            }

            // Propagate through remaining layers
            for (let i = 1; i < layers.length; i++) {
                layers[i].calculate();
            }
        },

        /**
         * Backpropagation: compute gradients from output error
         *
         * Computes gradient of loss with respect to each weight by
         * propagating errors backward through the network using chain rule.
         *
         * Process:
         *   1. Start with output layer error: δ^L
         *   2. For each layer L to 1 (backward):
         *      a. Compute weight gradients: ∂Loss/∂W^l
         *      b. Propagate error to previous layer: δ^(l-1)
         *
         * The error signal δ^l represents ∂Loss/∂z^l (gradient w.r.t. pre-activation)
         *
         * @param {Array<number>} error - Error gradient from loss function (δ^L)
         */
        gradient: function(error) {
            // Copy error values to last layer (avoid reference replacement)
            const lastLayer = layers[size.length-1];
            for (let i = 0; i < error.length; i++) {
                lastLayer.error[i] = error[i];
            }

            // Backpropagate through all layers (excluding input layer)
            for (let i = size.length-1; i > 0; i--) {
                layers[i].gradient();
            }
        },
        resetGradient: function() {
            // Reset the gradient
            for (let i = 1; i < layers.length; i++) {
                layers[i].resetGradient();
            }
        },
        updateWeights: function(learningRate, batchSize) {
            // Update weights for all layers (skip layer 0 - input layer has no weights)
            for (let i = 1; i < layers.length; i++) {
                layers[i].updateWeights(learningRate, batchSize);
            }
        },
        /**
         * Get all network weights
         *
         * Extracts weights from all layers (excluding input layer which has no weights).
         * Returns a nested array structure that can be saved to disk.
         *
         * @returns {Array<Array<Array<number>>>} Weights structure:
         *   [layer][neuron][weight]
         */
        getWeights: function() {
            const weights = [];

            // Extract weights for each layer (skip layer 0 - input layer has no weights)
            for (let i = 1; i < layers.length; i++) {
                const layerWeights = [];

                // Extract weights for each neuron in this layer
                for (let j = 0; j < layers[i].neurons.length; j++) {
                    layerWeights.push(layers[i].neurons[j].weights);
                }

                weights.push(layerWeights);
            }

            return weights;
        },

        /**
         * Set all network weights
         *
         * Loads weights into the network. Each neuron's weight array is replaced
         * with the provided weights.
         *
         * @param {Array<Array<Array<number>>>} weights - Weights structure:
         *   [layer][neuron][weight]
         */
        setWeights: function(weights) {
            // Load weights for each layer (skip layer 0 - input layer has no weights)
            for (let i = 1; i < layers.length; i++) {
                const layerWeights = weights[i - 1]; // weights array starts at 0, layers start at 1

                // Load weights for each neuron in this layer
                for (let j = 0; j < layers[i].neurons.length; j++) {
                    layers[i].neurons[j].weights = layerWeights[j];
                }
            }
        }
    }
}


/** Application **/

// Hyperparameters
const LEARNING_RATE = 0.01;

// Normalization statistics (calculated from MNIST training set)
// MNIST pixel values are in [0, 1] after division by 255
const PIXEL_MEAN = 0.1307; // Mean pixel value for MNIST
const PIXEL_STD = 0.3081;  // Standard deviation for MNIST

/**
 * Normalize pixel values using z-score normalization (standardization)
 *
 * Formula: x_norm = (x - μ) / σ
 *
 * Where:
 *   - x = original pixel value [0, 1] (already divided by 255)
 *   - μ = mean of all training set pixels (0.1307 for MNIST)
 *   - σ = standard deviation of all training set pixels (0.3081 for MNIST)
 *   - x_norm = normalized value (typically in range [-2, 2])
 *
 * Benefits:
 *   - Centers data around 0 (zero mean)
 *   - Scales to unit variance (σ = 1)
 *   - Helps gradient descent converge faster
 *   - Prevents numerical instability
 *
 * @param {Array<number>} pixels - Raw pixel values [0, 1]
 * @returns {Array<number>} Normalized pixel values
 */
const normalizedBuffer = new Array(784);
function normalizePixels(pixels) {
    for (let i = 0; i < 784; i++) {
        normalizedBuffer[i] = (pixels[i] - PIXEL_MEAN) / PIXEL_STD;
    }
    return normalizedBuffer;
}

/**
 * Train network on a mini-batch of images
 *
 * This function implements mini-batch stochastic gradient descent (SGD):
 *   1. Reset accumulated gradients
 *   2. For each image in batch:
 *      a. Forward pass: compute predictions
 *      b. Compute loss
 *      c. Backward pass: accumulate gradients
 *   3. Update weights using averaged gradients
 *
 * Mini-batch SGD balances:
 *   - Batch GD: slow but stable (uses all data)
 *   - Online SGD: fast but noisy (uses one example)
 *   - Mini-batch: good balance (uses small batches)
 *
 * @param {Object} network - Neural network instance
 * @param {number} position - Batch index (multiplied by 10 for image positions)
 * @returns {Object} Training metrics: {loss, accuracy, correct, total}
 */
const run = function(network, position) {
    // Load batch of 10 images from MNIST dataset
    let images = readMNIST(position*10, position*10+10);

    let totalLoss = 0;
    let correctPredictions = 0;

    // Reset accumulated gradients from previous batch
    network.resetGradient();

    // Pre-allocate reusable arrays (performance optimization)
    const target = new Array(10).fill(0);  // One-hot encoded label
    const error = new Array(10);           // Error gradient

    // Process each image in the batch
    for (let imgIdx = 0; imgIdx < images.length; imgIdx++) {
        const image = images[imgIdx];

        // ========== STEP 1: PREPARE TARGET (ONE-HOT ENCODING) ==========
        // One-hot encoding converts label to vector
        // Example: label=3 → [0,0,0,1,0,0,0,0,0,0]
        // This represents the true probability distribution (100% for correct class)
        for (let i = 0; i < 10; i++) {
            target[i] = (i === image.label) ? 1 : 0;
        }

        // ========== STEP 2: NORMALIZE INPUT ==========
        // Z-score normalization: x_norm = (x - μ) / σ
        // Centers data around 0 and scales to unit variance
        const normalizedPixels = normalizePixels(image.pixels);

        // ========== STEP 3: FORWARD PASS ==========
        // Compute network output from input
        network.forward(normalizedPixels);

        // Apply softmax to get probability distribution
        // Softmax: p_i = exp(z_i) / Σ(exp(z_j)) for all j
        // Converts raw scores (logits) to probabilities that sum to 1
        const output = network.output(softmax);

        // ========== STEP 4: EVALUATE PREDICTION ==========
        // Find predicted class (argmax of output probabilities)
        let maxValue = output[0];
        let predictedLabel = 0;
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                predictedLabel = i;
            }
        }
        if (predictedLabel === image.label) {
            correctPredictions++;
        }

        // ========== STEP 5: COMPUTE LOSS ==========
        // Cross-Entropy Loss measures difference between predicted and true distributions
        // Formula: L = -Σ(target[i] * log(output[i])) for i=0 to 9
        // Since target is one-hot, only one term is non-zero:
        //   L = -log(output[correct_class])
        // Lower loss = better predictions
        totalLoss += crossEntropyLoss(output, target);

        // ========== STEP 6: COMPUTE OUTPUT GRADIENT ==========
        // Combined derivative of Softmax + Cross-Entropy Loss
        // Mathematical beauty: ∂L/∂z[i] = output[i] - target[i]
        //
        // Intuition:
        //   - If output[i] > target[i]: gradient is positive → decrease z[i]
        //   - If output[i] < target[i]: gradient is negative → increase z[i]
        //   - Magnitude = how wrong the prediction is
        //
        // This is why softmax + cross-entropy are used together!
        for (let i = 0; i < output.length; i++) {
            error[i] = output[i] - target[i];
        }

        // ========== STEP 7: BACKWARD PASS ==========
        // Backpropagate error through network to compute weight gradients
        // Gradients are accumulated (added) across all examples in batch
        network.gradient(error);
    }

    // ========== STEP 8: UPDATE WEIGHTS ==========
    // Apply gradient descent with averaged gradients
    // w = w - α * (1/m) * Σ(∇w)
    network.updateWeights(LEARNING_RATE, images.length);

    // Return performance metrics for this batch
    return {
        loss: totalLoss / images.length,              // Average loss
        accuracy: (correctPredictions / images.length) * 100, // Percentage correct
        correct: correctPredictions,                   // Number correct
        total: images.length                           // Batch size
    };
}

const main = function() {
    const filepath = './data.json';

    // Create the network
    const network = Network([784, 16, 16, 10], relu);

    const totalBatches = 10000;

    // Load saved state if it exists
    let startBatch = 0;
    if (fs.existsSync(filepath)) {
        try {
            const savedData = JSON.parse(fs.readFileSync(filepath, 'utf8'));
            network.setWeights(savedData.weights);
            startBatch = savedData.batchIndex || 0;
            console.log(`Loaded weights from ${filepath}`);
            console.log(`Resuming from batch ${startBatch}`);

            // Check if training is already complete
            if (startBatch >= totalBatches) {
                console.log('');
                console.log('='.repeat(70));
                console.log('TRAINING ALREADY COMPLETE');
                console.log('='.repeat(70));
                console.log(`All ${totalBatches} batches have been processed`);
                console.log(`To restart training, delete ${filepath} or set batchIndex to 0`);
                console.log('='.repeat(70));
                return;
            }
        } catch (e) {
            console.log(`Error loading saved data: ${e.message}`);
            console.log(`Starting fresh with random initialization`);
        }
    } else {
        console.log(`No saved data found at ${filepath}`);
        console.log(`Starting with random initialization`);
    }

    console.log('');
    console.log('='.repeat(70));
    console.log('MNIST TRAINING');
    console.log('='.repeat(70));
    console.log(`Architecture: [784, 16, 16, 10]`);
    console.log(`Learning rate: ${LEARNING_RATE}`);
    console.log(`Total batches: ${totalBatches}`);
    console.log(`Batch size: 10 images`);
    console.log(`Starting batch: ${startBatch}`);
    console.log(`Remaining batches: ${totalBatches - startBatch}`);
    console.log('='.repeat(70));
    console.log('');
    const reportInterval = 100; // Report every 100 batches
    let recentMetrics = [];
    let allMetrics = [];
    const startTime = Date.now();

    for (let i = startBatch; i < totalBatches; i++) {
        const batchStartTime = Date.now();
        const metrics = run(network, i);
        const batchTime = Date.now() - batchStartTime;

        const metricData = { ...metrics, batchTime };
        recentMetrics.push(metricData);
        allMetrics.push(metricData);

        // Report progress every reportInterval batches
        if ((i + 1) % reportInterval === 0) {
            const avgLoss = recentMetrics.reduce((sum, m) => sum + m.loss, 0) / recentMetrics.length;
            const avgAccuracy = recentMetrics.reduce((sum, m) => sum + m.accuracy, 0) / recentMetrics.length;
            const avgBatchTime = recentMetrics.reduce((sum, m) => sum + m.batchTime, 0) / recentMetrics.length;
            const elapsedTime = (Date.now() - startTime) / 1000;
            const overallSpeed = ((i + 1) * 10) / elapsedTime;
            const progress = ((i + 1) / totalBatches * 100).toFixed(1);
            const batchRange = `${i - reportInterval + 2}-${i + 1}`;

            console.log('');
            console.log('='.repeat(70));
            console.log(`SUMMARY - Batches ${batchRange} (Progress: ${progress}%)`);
            console.log('-'.repeat(70));
            console.log(`Avg Loss: ${avgLoss.toFixed(4)} | Avg Accuracy: ${avgAccuracy.toFixed(1)}%`);
            console.log(`Avg Batch Time: ${avgBatchTime.toFixed(0)}ms | Avg Speed: ${overallSpeed.toFixed(1)} img/s`);
            console.log(`Overall Speed: ${overallSpeed.toFixed(1)} img/s | Total Time: ${elapsedTime.toFixed(1)}s`);
            console.log('='.repeat(70));

            recentMetrics = [];
        }

        // Save every 500 batches
        if ((i + 1) % 500 === 0) {
            const saveData = {
                weights: network.getWeights(),
                batchIndex: i + 1
            };
            fs.writeFileSync(filepath, JSON.stringify(saveData, null, 2));
            console.log(`\nSaved progress at batch ${i + 1}`);
        }
    }

    const totalTime = (Date.now() - startTime) / 1000;
    const avgBatchTime = allMetrics.reduce((sum, m) => sum + m.batchTime, 0) / allMetrics.length;

    console.log('');
    console.log('');
    console.log('='.repeat(70));
    console.log('TRAINING COMPLETE');
    console.log('='.repeat(70));
    console.log(`Total images processed: ${(totalBatches - startBatch) * 10}`);
    console.log(`Total time: ${totalTime.toFixed(2)}s`);
    console.log(`Average speed: ${((totalBatches - startBatch) * 10 / totalTime).toFixed(1)} images/second`);
    console.log(`Average batch time: ${avgBatchTime.toFixed(0)}ms`);
    console.log('='.repeat(70));

    // Final save
    const saveData = {
        weights: network.getWeights(),
        batchIndex: totalBatches
    };
    fs.writeFileSync(filepath, JSON.stringify(saveData, null, 2));
    console.log(`\nFinal weights saved to ${filepath}`);
}

main();