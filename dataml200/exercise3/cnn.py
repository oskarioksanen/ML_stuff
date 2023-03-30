import tensorflow as tf
import numpy as np
import os
import PIL
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skimage.io import imread_collection
from skimage.io import imread
import os

def create_cnn():
    model = tf.keras.models.Functional()
    return model

def load_data(paths, file_spec):

    # Yritys imread_collection
    """images = []
    class_labels = []

    for class_path, label in paths:
        load_pattern = os.path.join(class_path, file_spec)
        class_im = imread_collection(load_pattern, load_func=)
        class_im = np.array(class_im, dtype=np.float32)
        print(class_im.shape)
    #images = np.array(images, dtype=np.float32)
    print(images.shape)
    print(images)"""

    data = []
    class_labels = []

    for class_path, label in class_paths:
        for file_name in os.listdir(class_path):
            if file_name.endswith(".jpg"):
                image_path = os.path.join(class_path, file_name)
                image = PIL.Image.open(image_path)
                im_array = np.array(image, dtype=np.float32)
                data.append(im_array)
                class_labels.append(label)

    data = np.array(data)
    data = data.reshape(-1, 3, 64, 64).astype("float32")
    class_labels = np.array(class_labels)

    train_data, test_data, train_labels, test_labels = train_test_split(data,
                                                                        class_labels,
                                                                        test_size=0.2,
                                                                        random_state=20,
                                                                        shuffle=True)

    # If we use CategorialCrossentropy instead of SparseCategorialCrossentropy
    # We need to change train labels to onehot arrays
    # train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)

    # Normalize our data
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    return [train_data, test_data, train_labels, test_labels]

# Main
class_paths = [(r'C:\Users\oksan\anaconda3\envs\tf\ex2_dataml200\GTSRB_subset_2\GTSRB_subset_2\class1', 0),
               (r'C:\Users\oksan\anaconda3\envs\tf\ex2_dataml200\GTSRB_subset_2\GTSRB_subset_2\class2', 1)]
file_spec = '*.jpg'

data = load_data(class_paths, file_spec)

train_data = data[0]
test_data = data[1]
train_labels = data[2]
test_labels = data[3]

# Prints for debugging
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print("Training labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)

cnn_model = create_cnn()
