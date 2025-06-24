import numpy as np
from network.neuron import Neuron

class Layer:
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def forward(self, inputs: list[float]) -> list[float]:
        return [neuron.forward(inputs) for neuron in self.neurons]
       
    def backward(self, d_outs: list[float]) -> list[float]:
        # Rückwärtsrichtung: Liste von Gradienten aus nächster Schicht
        # Addiere die zurückfließenden Gradienten pro Eingabewert
        input_grads = np.zeros_like(self.neurons[0].last_input)
        for d_out, neuron in zip(d_outs, self.neurons):
            input_grads += neuron.backward(d_out)
        return input_grads.tolist()

    def update_params(self, learning_rate: float):
        for neuron in self.neurons:
            neuron.update_params(learning_rate)

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()
