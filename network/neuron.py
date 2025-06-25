
import numpy as np


class Neuron():

    def __init__(self, weights: list[float], bias: float):
        # Weights and Bias
        self.weights = np.array(weights, dtype=float)
        self.bias = bias

        # Gradients (How much adjustion)
        self.delta_weights = np.zeros_like(self.weights)
        self.delta_bias = 0.0

        # Last Inputs & Pre-Activation
        self.last_input = None
        self.last_pre_actf = None

 # Forwards 
    def forward(self, inputs: list[float]) -> float:
        # Inputs X Weights + Bias -> Relu
        inputs = np.array(inputs, dtype=float)
        self.last_input = inputs # Save
        z = np.dot(inputs, self.weights) + self.bias
        self.last_pre_actf = z # Save
        return self.relu(z)


    def relu(self, x):
        return np.maximum(0, x)
    

 # Backwards
    def backward(self, d_out: float) -> np.ndarray:
        # Check which direction our Pre-Activation-Value would have in act (how much)
        drelu = self.relu_derivative(self.last_pre_actf)

        # Mult the change Direction and the amount
        dz = d_out * drelu

        # Saving Gradients
        self.delta_weights = dz * self.last_input # Bigger inputs mean bigger adjustion for the weights smaller mean less adjustion
        self.delta_bias = dz

        # Gradient für vorherige Schicht (falls verkettet)
        return dz * self.weights  # Bigger weights mean bigger adjustion for the inputs smaller mean less adjustion
    
    def relu_derivative(self, x: float) -> float:
        # ReLU direction either 0 (Stable) or 1 (f(x)=x derivative)
        return 1.0 if x > 0 else 0.0
    

 # Additional
    def update_params(self, learning_rate: float):
        # Update weights and biases based on presaved gradients and the curr learning rate
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias

    def zero_grad(self):
        # Reset Gradients
        self.delta_bias = np.zeros_like(self.weights) # [1.33, 4.23, 43.44] -> [0,0,0]
        self.delta_weights = 0.0
    
    


        