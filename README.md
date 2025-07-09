# 🧠 MNIST Neural Network from Scratch (NumPy-only)
This repository contains a fully from-scratch neural network implementation using only Python and NumPy, applied to the MNIST handwritten digits dataset.

⚠️ **Note:** This project uses an educational design with individual Neuron objects per layer. While this drastically reduces performance and training speed, it greatly increases intuitiveness, readability, and ease of fine-tuning for learning purposes.

## 📁 Project Structure

```
MNIST_CUSTOM/
├── data/
│   ├── train/               # Training images (MNIST format)
│   └── test/                # Test images (MNIST format)
│
├── models/
│   └── 40k_v1_model.pkl     # Pre-trained model (40k parameters)
│
├── network/                 # Core neural network logic (built from scratch)
│   ├── __init__.py
│   ├── neuron.py            # Neuron class (with forward/backward passes)
│   ├── layer.py             # Layer class (collection of neurons)
│   └── neural_network.py    # Complete network (built from layers)
│
├── training/
│   ├── config.py            # 🔧 Adjustable training parameters (epochs, LR, decay...)
│   ├── data_loader.py       # Handles loading and preprocessing of MNIST data
│   └── train.py             # Main training logic
│
├── testing/
│   └── test.py              # Script to evaluate a trained model
│
├── main.py                  # Run this to test the model on custom images
└── README.md                # This file
```

## 🧩 Key Concepts
### Neurons as Objects 
Each Neuron is implemented as a Python class with individual weights, biases, activation (ReLU), gradients, and update logic.

### Layer
A Layer is a simple container of Neurons that performs forward/backward propagation in sequence.

### Neural Network
The NeuralNetwork class combines layers, manages forward and backward passes, and updates parameters.

### No Frameworks
No TensorFlow, PyTorch or high-level abstractions. This is bare-metal NumPy to showcase what’s under the hood of deep learning.

## 🚀 Getting Started
### 1. Clone the repository
```bash
git clone https://github.com/your-username/MNIST-NeuralNet-Scratch.git
cd MNIST-NeuralNet-Scratch
```
### 2. Install Requirements
Only numpy (and pickle for saving/loading models) are needed:

```bash
pip install pickle
pip install numpy
```
### 3. Train the Network
You can configure your training run in training/config.py (e.g., learning rate, epochs, decay):

```python
learning_rate = 0.01
epochs = 15
batch_size = 64
decay = 0.99
```
Then run:

```bash
python training/train.py
```
### 4. Test an Existing Model
A pre-trained model is included (models/40k_v1_model.pkl):

```bash
python testing/test.py
```
Or run the model on an image via:

```bash
python main.py
```
## 🧠 Example: Neuron Class
```python
z = np.dot(inputs, self.weights) + self.bias
if self.use_relu:
    return self.relu(z)
return z
```
Each neuron tracks its last input and gradient, and applies ReLU conditionally for backpropagation.
