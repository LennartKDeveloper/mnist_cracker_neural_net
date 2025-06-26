import sys
import os

# Füge das Parent-Verzeichnis von "training" dem Suchpfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import math
import pickle
import numpy as np
from random import uniform

from network.layer import Layer
from network.neural_network import Network
from network.neuron import Neuron
from training.data_loader import DataLoader


input_size = 784  # 28x28 Bilder
hidden_size = 64
output_size = 10  # 10 Klassen für Ziffern 0–9

# RELU based varity in Weights and bias initialisation
def he_weights(n_inputs):
    stddev = np.sqrt(2.0 / n_inputs)
    return (np.random.randn(n_inputs) * stddev).tolist()


layer1 = Layer([Neuron(he_weights(input_size), np.random.randn() * 0.01, True) for _ in range(hidden_size)],) # Hidden
layer2 = Layer([Neuron(he_weights(hidden_size), np.random.randn() * 0.01, True) for _ in range(hidden_size)],) # Hidden
layer3 = Layer([Neuron(he_weights(hidden_size), np.random.randn() * 0.01, False) for _ in range(output_size)],) # Output

network = Network([layer1, layer2, layer3])

# Train Data
train_size = 40000 # 0 - 59999
training_input_images = DataLoader().load_mnist_train_data()["images"][:train_size] / 255.0 # Array 0.0 - 1.0
training_output_labels = DataLoader().load_mnist_train_data()["labels"][:train_size]

learning_rate = 0.001
epochs = 5
decay_rate = 0.95
drift_tracker = 0 

for epoch in range(epochs):
    total_loss = 0
    for img, y in zip(training_input_images, training_output_labels):
        x = img.flatten()  # 28x28 -> 784 Vektor
        output = network.forward(x)

        # print(f"Activation: Max: {np.max(output)} Min {np.min(output)}, avg: {np.average(output)}")
        # OneHot vector as target
        target = np.zeros(output_size)
        target[y] = 1.0

        loss = network.cross_entropy(output, target)
        if math.isnan(loss): drift_tracker+=1 # Nan counter 

        total_loss += loss
        network.zero_grad()
        network.backward(target)
        network.update_params(learning_rate)
        
    avg_loss = total_loss / len(training_input_images)
    print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")
    learning_rate *= decay_rate

print(f"Inf Drift: {drift_tracker}")


# Ask after training if user wants to save the model:
while True:
    save_input = input("Do you want to save the trained model? (yes/no): ").strip().lower()
    if save_input == "yes" or save_input == "y":
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(network, f)
        print("Model saved to 'trained_model.pkl'.")
        break
    elif save_input == "no" or save_input == "n":
        print("Model not saved. Program terminated.")
        break
    else:
        print("Please answer with 'yes' or 'no'.")
