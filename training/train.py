import sys
import os

# Get path access to train data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pickle
import numpy as np
from random import uniform

from network.layer import Layer
from network.neural_network import Network
from network.neuron import Neuron
from training.data_loader import DataLoader
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TRAIN_SIZE, LEARNING_RATE, EPOCHS, DECAY_RATE, BIAS_STDDEV


# RELU based variety in Weights and bias initialisation
def he_weights(n_inputs):
    stddev = np.sqrt(2.0 / n_inputs)
    return (np.random.randn(n_inputs) * stddev).tolist()


layer1 = Layer([Neuron(he_weights(INPUT_SIZE), np.random.randn() * BIAS_STDDEV, True) for _ in range(HIDDEN_SIZE)])
layer2 = Layer([Neuron(he_weights(HIDDEN_SIZE), np.random.randn() * BIAS_STDDEV, True) for _ in range(HIDDEN_SIZE)])
layer3 = Layer([Neuron(he_weights(HIDDEN_SIZE), np.random.randn() * BIAS_STDDEV, False) for _ in range(OUTPUT_SIZE)])

network = Network([layer1, layer2, layer3])

# Train Data
training_input_images = DataLoader().load_mnist_train_data()["images"][:TRAIN_SIZE] / 255.0
training_output_labels = DataLoader().load_mnist_train_data()["labels"][:TRAIN_SIZE]

learning_rate = LEARNING_RATE
drift_tracker = 0 

for epoch in range(EPOCHS):
    total_loss = 0
    for img, y in zip(training_input_images, training_output_labels):
        x = img.flatten()  # 28x28 -> 784 Vektor
        output = network.forward(x)

        target = np.zeros(OUTPUT_SIZE)
        target[y] = 1.0

        loss = network.cross_entropy(output, target)
        if math.isnan(loss):
            drift_tracker += 1

        total_loss += loss
        network.zero_grad()
        network.backward(target)
        network.update_params(learning_rate)
        
    avg_loss = total_loss / len(training_input_images)
    print(f"Epoch {epoch + 1}/{EPOCHS} — Loss: {avg_loss:.4f}")
    learning_rate *= DECAY_RATE

print(f"Inf Drift: {drift_tracker}")

# Ask after training if user wants to save the model:
while True:
    save_input = input("Do you want to save the trained model? (yes/no): ").strip().lower()
    if save_input in ["yes", "y"]:
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(network, f)
        print("Model saved to 'trained_model.pkl'.")
        break
    elif save_input in ["no", "n"]:
        print("Model not saved. Program terminated.")
        break
    else:
        print("Please answer with 'yes' or 'no'.")
