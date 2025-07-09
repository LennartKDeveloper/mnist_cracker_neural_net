import sys
import os

# Füge das Parent-Verzeichnis von "training" dem Suchpfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pickle
import numpy as np
from training.data_loader import DataLoader

def load_model(path: str = "trained_model.pkl"):
    # Load a model from a pickle file.
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(model, image: np.ndarray) -> int:
    # Predicted class for a single image using the provided model.
    if image.ndim == 2:
        image = image.flatten()
    output = model.forward(image)
    return int(np.argmax(output))

def evaluate_model(model, images: np.ndarray, labels: np.ndarray) -> float:
    # Calculates the accuracy of the model on a testset of images and labels.
    correct = 0
    for img, label in zip(images, labels):
        prediction = predict(model, img)
        if prediction == label:
            correct += 1
    return correct / len(images) * 100


# Load MNIST-Testdata 
test_data = DataLoader().load_mnist_test_data()
images = test_data["images"] / 255.0  # Normalisierung
labels = test_data["labels"]

# Load Modell
model = load_model("models/40k_v1_model.pkl")

# Calculate Accuracy
accuracy = evaluate_model(model, images, labels)
print(f"Test Accuracy: {accuracy:.2f}%")



