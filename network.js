import fs from 'fs';
import {
    crossEntropyLoss,
    relu,
    softmax
} from './utils.js';
import {readMNIST} from "./images.js";

const initializeWeights = (numInputs) => {
    // He initialization with Gaussian distribution
    // Uses Box-Muller transform for true normal distribution
    const stdDev = Math.sqrt(2 / numInputs);
    const weights = [];

    for (let i = 0; i < numInputs; i++) {
        // Box-Muller transform to generate Gaussian random numbers
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        weights[i] = z * stdDev;
    }

    return weights;
}

const sumAllElements = function(A, B) {
    let sum = 0;
    for (let i = 0; i < A.length; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

const Neuron = function(previousLayer) {
    let size = previousLayer.value.length;

    return {
        value: null,
        raw: null,
        input: previousLayer.value,
        weights: initializeWeights(size),
        calculate: function(activation) {
            // GPU dot product
            this.value = this.raw = sumAllElements(this.input, this.weights);
            // Apply activation function
            if (activation) {
                this.value = activation(this.raw);
            }
            return this.value;
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
        updateWeights: function(learningRate, batchSize) {
            // Update weights for all neurons in this layer
            for (let i = 0; i < numOfNeurons; i++) {
                let weights = neurons[i].weights;
                for (let j = 0; j < weights.length; j++) {
                    // Average the gradient
                    const avgGradient = weightGradient[i][j] / batchSize;
                    // Gradient descent: w_new = w_old - learning_rate * gradient
                    weights[j] -= learningRate * avgGradient;
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

        layer.gradient = function() {
            // Initialize previous layer errors to zero
            for (let j = 0; j < e.length; j++) {
                e[j] = 0;
            }

            // Loop through all neurons
            for (let i = 0; i < numOfNeurons; i++) {
                for (let j = 0; j < numOfInputs; j++) {
                    // Part 1: Weight gradient for THIS layer
                    weightGradient[i][j] += this.error[i] * v[j];
                    // Part 2: Backpropagate error (ALWAYS add, no ReLU yet)
                    e[j] += neurons[i].weights[j] * this.error[i];
                }
            }

            // Part 3: Apply ReLU derivative AFTER accumulation (only if previous layer has neurons)
            if (n) {
                for (let j = 0; j < n.length; j++) {
                    if (n[j].raw <= 0) {
                        e[j] = 0;
                    }
                }
            }
        }

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

const Network = function(size, activationMethod) {
    const layers = [];

    for (let i = 0; i < size.length; i++) {
        // Activation
        let activation = i < size.length - 1 ? activationMethod : null;
        // New layer
        layers.push(Layer(size[i], layers[i-1], activation))
    }

    return {
        layers: layers,
        output: function(convert) {
            let output = layers[size.length-1].value;
            if (convert) {
                // Apply softmax for example
                output = convert(output);
            }
            return output;
        },
        forward: function(input) {
            // Copy input values to first layer
            for (let i = 0; i < input.length; i++) {
                layers[0].value[i] = input[i];
            }
            // Forward pass through all layers
            for (let i = 1; i < layers.length; i++) {
                layers[i].calculate();
            }
        },
        gradient: function(error) {
            // Copy error values to last layer instead of replacing reference
            const lastLayer = layers[size.length-1];
            for (let i = 0; i < error.length; i++) {
                lastLayer.error[i] = error[i];
            }

            // Backward pass through all layers
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
        load: function(filepath) {
            // Load weights from file if it exists
            if (fs.existsSync(filepath)) {
                const weights = JSON.parse(fs.readFileSync(filepath, 'utf8'));

                // Load weights for each layer (skip layer 0 - input layer has no weights)
                for (let i = 1; i < layers.length; i++) {
                    const layerWeights = weights[i - 1]; // weights array starts at 0, layers start at 1

                    // Load weights for each neuron in this layer
                    for (let j = 0; j < layers[i].neurons.length; j++) {
                        layers[i].neurons[j].weights = layerWeights[j];
                    }
                }

                console.log(`Weights loaded from ${filepath}`);
            } else {
                console.log(`No saved weights found at ${filepath}, using random initialization`);
            }
        },
        save: function(filepath) {
            // Save only weights to file
            const weights = [];

            // Save weights for each layer (skip layer 0 - input layer has no weights)
            for (let i = 1; i < layers.length; i++) {
                const layerWeights = [];

                // Save weights for each neuron
                for (let j = 0; j < layers[i].neurons.length; j++) {
                    layerWeights.push(layers[i].neurons[j].weights);
                }

                weights.push(layerWeights);
            }

            fs.writeFileSync(filepath, JSON.stringify(weights, null, 2));
            console.log(`Weights saved to ${filepath}`);
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
 * Normalize pixel values using z-score normalization
 * Formula: (x - mean) / std
 * This centers the data around 0 and scales to unit variance
 */
const normalizedBuffer = new Array(784);
function normalizePixels(pixels) {
    for (let i = 0; i < pixels.length; i++) {
        normalizedBuffer[i] = (pixels[i] - PIXEL_MEAN) / PIXEL_STD;
    }
    return normalizedBuffer;
}

// Run the network
const run = function(network, position) {
    // Load batch of 10 images from MNIST dataset
    let images = readMNIST(position*10, position*10+10);

    let totalLoss = 0;
    let correctPredictions = 0;

    // Reset gradient
    network.resetGradient();

    // Pre-allocate reusable arrays
    const target = new Array(10).fill(0);
    const error = new Array(10);

    // Process each image in the batch
    images.forEach(function (image) {
        // Reset and set one-hot encoded target vector
        // Example: if label=3, target = [0,0,0,1,0,0,0,0,0,0]
        for (let i = 0; i < 10; i++) {
            target[i] = (i === image.label) ? 1 : 0;
        }

        // ========== NORMALIZATION ==========
        // Formula: x_normalized = (x - μ) / σ
        // Where: μ = mean (0.1307), σ = std deviation (0.3081)
        // This centers data around 0 and scales to unit variance
        const normalizedPixels = normalizePixels(image.pixels);

        // Run the batch
        network.forward(normalizedPixels);

        // Get the output with softmax
        const output = network.output(softmax);

        // Check if prediction is correct
        // Prediction is the index with the highest probability
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

        // Calculate Cross-Entropy Loss
        // Formula: L = -Σ(target[i] * log(output[i])) for i=0 to 9
        // Since target is one-hot, this simplifies to: L = -log(output[correct_class])
        totalLoss += crossEntropyLoss(output, target);

        // OUTPUT LAYER GRADIENT
        // Combined derivative of Softmax and Cross-Entropy Loss
        // Formula: ∂L/∂z2[i] = output[i] - target[i]
        // This is a beautiful simplification! When using softmax + cross-entropy together,
        // the derivative is just (prediction - true_label)
        for (let i = 0; i < output.length; i++) {
            error[i] = output[i] - target[i];
        }

        // Calculate the gradient
        network.gradient(error);
    });

    // Update weights after processing the batch
    network.updateWeights(LEARNING_RATE, images.length);

    // Return metrics
    return {
        loss: totalLoss / images.length,
        accuracy: (correctPredictions / images.length) * 100,
        correct: correctPredictions,
        total: images.length
    };
}

const main = function() {
    // Create the network
    const network = Network([784, 16, 16, 10], relu);
    // Load weights in case exist
    network.load('./data.json');

    console.log('='.repeat(70));
    console.log('MNIST TRAINING');
    console.log('='.repeat(70));
    console.log(`Architecture: [784, 16, 16, 10]`);
    console.log(`Learning rate: ${LEARNING_RATE}`);
    console.log(`Total batches: 1000`);
    console.log(`Batch size: 10 images`);
    console.log('='.repeat(70));
    console.log('');

    const totalBatches = 1000;
    const reportInterval = 100; // Report every 100 batches
    let recentMetrics = [];
    let allMetrics = [];
    const startTime = Date.now();

    for (let i = 0; i < totalBatches; i++) {
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
            network.save('./data.json');
        }
    }

    const totalTime = (Date.now() - startTime) / 1000;
    const avgBatchTime = allMetrics.reduce((sum, m) => sum + m.batchTime, 0) / allMetrics.length;

    console.log('');
    console.log('');
    console.log('='.repeat(70));
    console.log('TRAINING COMPLETE');
    console.log('='.repeat(70));
    console.log(`Total images processed: ${totalBatches * 10}`);
    console.log(`Total time: ${totalTime.toFixed(2)}s`);
    console.log(`Average speed: ${(totalBatches * 10 / totalTime).toFixed(1)} images/second`);
    console.log(`Average batch time: ${avgBatchTime.toFixed(0)}ms`);
    console.log('='.repeat(70));

    // Final save
    network.save('./data.json');
}

main();