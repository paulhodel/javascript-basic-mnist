# MNIST Neural Network

A simple neural network implementation in JavaScript for recognizing handwritten digits from the MNIST dataset.

## Overview

This project implements a basic feedforward neural network from scratch (no ML libraries) to classify handwritten digits (0-9). The network uses:

- **Architecture**: 784 input neurons → 16 hidden neurons → 16 hidden neurons → 10 output neurons
- **Activation**: ReLU for hidden layers, Softmax for output
- **Loss**: Cross-entropy loss
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Training**: Mini-batch training with batch size of 10

## Getting Started

### Clone the repository

```bash
git clone https://github.com/paulhodel/javascript-basic-mnist.git
cd ai
```

### Install dependencies

```bash
npm install
```

### Train the model

```bash
node index.js
```

This will train the network for 1000 batches and save the weights to `w.json`. You'll see progress output showing loss and accuracy per batch.

### Test the model

```bash
node test.js
```

This will evaluate the trained model on test images and show predictions.

## Project Structure

- `index.js` - Main training script
- `images.js` - MNIST dataset loader
- `utils.js` - Neural network utility functions (forward pass, activation functions, etc.)
- `w.json` - Saved model weights (generated after training)

## Example Output

```
Batch 0 - Loss: 2.3026, Accuracy: 10.0% (1/10)
Batch 1 - Loss: 2.2891, Accuracy: 20.0% (2/10)
...
============================================================
Progress: 10/1000 (1.0%)
Last 10 batches - Avg Loss: 2.1234, Avg Accuracy: 35.5%
Loss Range: 2.0123 - 2.2456
============================================================
```

## Future Improvements

- Bias terms
- Batch normalization
- Dropout regularization
- Adam optimizer
- Learning rate scheduling
- Data augmentation
