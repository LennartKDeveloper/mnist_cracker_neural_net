
import numpy as np

class Network:

    def __init__(self, layers):
        self.layers = layers  # Liste von Layer-Objekten
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
            # print(inputs[0])
        
        self.last_output = self.softmax(np.array(inputs))
        # print(f"Softmax: {self.last_output}")
        return self.last_output

    def backward(self, target: list[float]):
        # target: one-hot encoded Vektor, z. B. [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] für Klasse 2

        # Softmax + Cross Entropy Ableitung (vereinfachte Form):
        # dL/dz = y_hat - y (y_hat = softmax output, y = one-hot)
        d_loss = self.last_output - np.array(target)

        # Backpropagation durch die Layer in umgekehrter Reihenfolge
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss)

    def update_params(self, learning_rate):
        for layer in self.layers:
            layer.update_params(learning_rate)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))  # numerisch stabil
        return exp_logits / np.sum(exp_logits)

    def cross_entropy(self, prediction: list[float], target: list[float]) -> float:
        # prediction: Softmax-Ausgabe
        # target: One-Hot
        epsilon = 1e-12  # für numerische Stabilität
        prediction = np.clip(prediction, epsilon, 1. - epsilon)
        return -np.sum(np.array(target) * np.log(prediction))


    

    