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
from keras.utils.vis_utils import plot_model

class TrafficSignIdentifier(tf.keras.Model):
    def __init__(self):
        super(TrafficSignIdentifier, self).__init__()
        model_inputs = tf.keras.Input(shape=(64, 64, 3))
        conv2D_1 = tf.keras.layers.Conv2D(
                                    filters=10,
                                    kernel_size=(3,3),
                                    strides=2,
                                    activation='relu',
                                    padding='same',
                                    input_shape=(64, 64, 3)
                                    )(model_inputs)
        max_pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2D_1)
        conv2D_2 = tf.keras.layers.Conv2D(
                                    filters=10,
                                    kernel_size=(3,3),
                                    strides=2,
                                    activation='relu',
                                    padding='same',
                                    input_shape=(64, 64, 3)
                                    )(max_pooling_1)
        max_pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2D_2)
        flatten = tf.keras.layers.Flatten()(max_pooling_2)
        model_outputs = tf.keras.layers.Dense(2, activation='sigmoid')(flatten)
        self.model = tf.keras.Model(inputs=model_inputs,
                                    outputs=model_outputs,
                                    name="GTSRB_model")
        self.model.summary()

    def call(self, x):
        identifier = self.model(x)
        return identifier

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
    #data = data.reshape(-1, 3, 64, 64).astype("float32")
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
class_paths = [(r'C:\Users\oksan\anaconda3\envs\dataml_200\ex2\GTSRB_subset_2\GTSRB_subset_2\class1', 0),
               (r'C:\Users\oksan\anaconda3\envs\dataml_200\ex2\GTSRB_subset_2\GTSRB_subset_2\class2', 1)]
file_spec = '*.jpg'

data = load_data(class_paths, file_spec)

train_data = data[0]
test_data = data[1]
train_labels = data[2]
# If we use tf.keras.losses.BinaryCrossentropy(from_logits=True) as
# loss function
#train_labels = tf.one_hot(train_labels, 2)
test_labels = data[3]

# Prints for debugging
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print("Training labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)

ts_identifier = TrafficSignIdentifier()

#loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
ts_identifier.compile(optimizer='SGD',
                      loss=loss_fn,
                     metrics=['accuracy'])
history = ts_identifier.fit(train_data, train_labels,
                 epochs=20,
                 shuffle=True,
                 batch_size=32,
                 validation_data=(test_data, test_labels))


one_hot_preds = ts_identifier.predict(test_data)
preds = np.argmax(one_hot_preds, axis=1)
accuracy = accuracy_score(preds, test_labels)
print()
print('----------------')
print(f'Accuracy of CNN: {accuracy:.2f}')
print('----------------')
print()
#evaluations = ts_identifier.evaluate(test_data, test_labels)
