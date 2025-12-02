import fs from 'fs';
import { GPU } from 'gpu.js';
import {
    crossEntropyLoss,
    relu,
    softmax
} from './utils.js';
import {readMNIST} from "./images.js";

Array.prototype.copy = function(target) {
    if (!Array.isArray(target)) {
        throw new TypeError("Target must be an array");
    }
    if (target.length !== this.length) {
        throw new RangeError("Arrays must be the same length");
    }

    for (let i = 0; i < this.length; i++) {
        target[i] = this[i];
    }

    return target; // optional: returns the modified target
};

const initializeWeights = (numInputs) => {
    // Returns an array of random weights for one neuron
    const stdDev = Math.sqrt(2 / numInputs);
    return new Array(numInputs).fill(0).map(() =>
        (Math.random() * 2 - 1) * stdDev
    );
}

const sumAllElements = function(A, B) {
    let sum = 0;
    for (let i = 0; i < A.length; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

const Neuron = function(previousLayer) {
    let size = previousLayer.length;

    return {
        value: null,
        raw: null,
        input: previousLayer.value,
        weights: initializeWeights(size),
        calculate: function(activation) {
            // Calculate value of this neuron based on the input and its weights
            this.value = this.raw = sumAllElements(this.input, this.weights);
            // New value for this neuron
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
    if (previousLayer) {
        const numInputs = previousLayer.value.length;
        for (let i = 0; i < numOfNeurons; i++) {
            weightGradient[i] = new Array(numInputs).fill(0);
        }
    }

    return {
        // Cache curren value of the neurons on this network
        value: [],
        error: [],
        neurons: neurons,
        activation: activation,
        calculate: function() {
            // Calculate all neurons
            for (let j = 0; j < neurons.length; j++) {
                this.value[j] = neurons[j].calculate(activation);
            }
        },
        gradient: function() {
            if (previousLayer) {
                // Initialize previous layer errors
                for (let j = 0; j < previousLayer.value.length; j++) {
                    previousLayer.error[j] = 0;
                }

                // Loop through all neurons
                for (let i = 0; i < numOfNeurons; i++) {
                    for (let j = 0; j < previousLayer.value.length; j++) {
                        // Part 1: Weight gradient for THIS layer
                        weightGradient[i][j] += this.error[i] * previousLayer.value[j];
                        // Part 2: Backpropagate error (ALWAYS add, no ReLU yet)
                        previousLayer.error[j] += neurons[i].weights[j] * this.error[i];
                    }
                }

                // Part 3: Apply ReLU derivative AFTER accumulation
                for (let j = 0; j < previousLayer.neurons.length; j++) {
                    if (previousLayer.neurons[j].raw <= 0) {
                        previousLayer.error[j] = 0;
                    }
                }
            }
        },
        resetGradient: function() {
            for (let i = 0; i < numOfNeurons; i++) {
                for (let j = 0; j < previousLayer.value.length; j++) {
                    weightGradient[i][j] = 0;
                }
            }
        },
        updateWeights: function(learningRate, batchSize) {
            // Update weights for all neurons in this layer
            for (let i = 0; i < numOfNeurons; i++) {
                for (let j = 0; j < neurons[i].weights.length; j++) {
                    // Average the gradient
                    const avgGradient = weightGradient[i][j] / batchSize;

                    // Gradient descent: w_new = w_old - learning_rate * gradient
                    neurons[i].weights[j] -= learningRate * avgGradient;
                }
            }
        }
    }
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
            // Copy the values to the first layer: input layer
            layers[0].value.copy(input);
            // Forward
            for (let i = 1; i < layers.length; i++) {
                layers[i].calculate();
            }
        },
        gradient: function(error) {
            layers[size.length-1].error = error;

            // Forward
            for (let i = size.length-1; i > 0; i--) {
                layers[i].gradient();
            }
        },
        resetGradient: function() {
            // Reset the gradient
            for (let i = 0; i < layers.length; i++) {
                layers[i].resetGradient();
            }
        },
        updateWeights: function(learningRate, batchSize) {
            // Update weights for all layers (skip layer 0 - input layer has no weights)
            for (let i = 1; i < layers.length; i++) {
                layers[i].updateWeights(learningRate, batchSize);
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
 * Normalize pixel values using z-score normalization
 * Formula: (x - mean) / std
 * This centers the data around 0 and scales to unit variance
 */
function normalizePixels(pixels) {
    return pixels.map(pixel => (pixel - PIXEL_MEAN) / PIXEL_STD);
}

// Create the network
const network = Network([784, 16, 16, 10], relu);

// Run the network
const run = function(w, position) {
    // Load batch of 10 images from MNIST dataset
    let images = readMNIST(position*10, position*10+10);

    let totalLoss = 0;
    let correctPredictions = 0;

    // Reset gradient
    network.resetGradient();

    // Process each image in the batch
    images.forEach(function (image) {
        // Create one-hot encoded target vector
        // Example: if label=3, target = [0,0,0,1,0,0,0,0,0,0]
        let target = new Array(10).fill(0);
        target[image.label] = 1;

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
        const predictedLabel = output.indexOf(Math.max(...output));
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
        let error = output.map((prob, i) => prob - target[i]);

        // Calculate the gradient
        network.gradient(error);
    });

    // Update weights after processing the batch
    network.updateWeights(LEARNING_RATE, images.length);
}



