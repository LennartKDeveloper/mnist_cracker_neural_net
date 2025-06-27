import numpy as np
import struct


class DataLoader():

  def read_images(self, filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images

  def read_labels(self, filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
  def load_mnist_train_data(self):
    # load Train data in /data/train/
    train_images = self.read_images("data/train/train-images.idx3-ubyte") # 0- 59999
    train_labels = self.read_labels("data/train/train-labels.idx1-ubyte") # 0- 59999
    return {"images": train_images, "labels": train_labels}
  
  def load_mnist_test_data(self):
    # load Test data in /data/test/
    train_images = self.read_images("data/test/t10k-images.idx3-ubyte") # 0- 59999
    train_labels = self.read_labels("data/test/t10k-labels.idx1-ubyte") # 0- 59999
    return {"images": train_images, "labels": train_labels}


