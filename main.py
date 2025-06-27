import pickle
import numpy as np

from training.data_loader import DataLoader

def predict_image(image: np.ndarray, model_path: str = "trained_model.pkl") -> int:
    """
    Lädt das gespeicherte Modell und gibt die Vorhersage für ein einzelnes Bild aus.

    :param image: Ein flaches oder 28x28 NumPy-Array mit Werten zwischen 0 und 1.
    :param model_path: Pfad zur gespeicherten Pickle-Datei.
    """

    # Image flatten if it is 2D
    if image.ndim == 2:
        image = image.flatten()

    # load the model from the pickle file
    with open(model_path, "rb") as f:
        network = pickle.load(f)

    # Clculate the output of the network
    output = network.forward(image) # Output: list[float]

    # Print the predicted class and softmax output
    predicted_class = int(np.argmax(output))
    print(f"Predicted class: {predicted_class}")
    print(f"Softmax output: {output}")
    return predicted_class

def show_image(predicted_class: int, image: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis('off')
    plt.show()
        
        

if __name__ == "__main__":
  image_num = 7  # Index of the image to predict

  test_image = DataLoader().load_mnist_test_data()["images"][image_num] / 255.0
  pred = predict_image(test_image, "models/40k_v1_model.pkl")
  # optional: Show the image with prediction
  show_image(pred, test_image)  
