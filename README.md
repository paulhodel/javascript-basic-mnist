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
Batch 990 - Loss: 0.0344, Accuracy: 100.0% (10/10)
Batch 991 - Loss: 0.1439, Accuracy: 100.0% (10/10)
Batch 992 - Loss: 0.2650, Accuracy: 90.0% (9/10)
Batch 993 - Loss: 0.2535, Accuracy: 100.0% (10/10)
Batch 994 - Loss: 0.6497, Accuracy: 80.0% (8/10)
Batch 995 - Loss: 0.2757, Accuracy: 100.0% (10/10)
Batch 996 - Loss: 0.1965, Accuracy: 100.0% (10/10)
Batch 997 - Loss: 0.6101, Accuracy: 80.0% (8/10)
Batch 998 - Loss: 0.2125, Accuracy: 90.0% (9/10)
Batch 999 - Loss: 0.2009, Accuracy: 100.0% (10/10)
============================================================
Progress: 1000/1000 (100.0%)
Last 10 batches - Avg Loss: 0.2842, Avg Accuracy: 94.0%
Loss Range: 0.0344 - 0.6497
============================================================
```

## Future Improvements

- Bias terms
- Batch normalization
- Dropout regularization
- Adam optimizer
- Learning rate scheduling
- Data augmentation
