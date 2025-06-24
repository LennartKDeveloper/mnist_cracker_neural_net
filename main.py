import numpy as np

from data_loader import DataLoader
from network.layer import Layer
from network.neural_network import Network


from random import uniform

from network.neuron import Neuron


input_size = 784  # 28x28 Bilder
hidden_size = 64
output_size = 10  # 10 Klassen für Ziffern 0–9

# Dummy-Initialisierung
def random_weights(n): return [uniform(-0.1, 0.1) for _ in range(n)]

layer1 = Layer([Neuron(random_weights(input_size), 0.0) for _ in range(input_size)]) # Input
layer2 = Layer([Neuron(random_weights(input_size), 0.0) for _ in range(hidden_size)]) # Hidden
layer3 = Layer([Neuron(random_weights(hidden_size), 0.0) for _ in range(output_size)]) # Output

network = Network([layer1, layer2, layer3])

# 
training_input_images = DataLoader().load_mnist_train_data()["images"][:1000]
training_output_labels = DataLoader().load_mnist_train_data()["labels"][:1000]

learning_rate = 0.01
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for img, y in zip(training_input_images, training_output_labels):
        x = img.flatten()  # 28x28 -> 784 Vektor
        print("Neues Bild")
        output = network.forward(x)
        loss = network.cross_entropy(output, y)
        total_loss += loss

        network.zero_grad()
        network.backward(y)
        network.update_params(learning_rate)
    print("Neue Epoche")
        

    avg_loss = total_loss / len(training_input_images)
    print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")
