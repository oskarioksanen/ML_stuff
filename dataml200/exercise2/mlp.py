import tensorflow as tf
import numpy as np
import os
import PIL
from sklearn.model_selection import train_test_split
# For debugging
import matplotlib.pyplot as plt

def load_data(class_paths):

    data = []
    class_labels = []

    for class_path, label in class_paths:
        for file_name in os.listdir(class_path):
            if file_name.endswith(".jpg"):
                image_path = os.path.join(class_path, file_name)
                image = PIL.Image.open(image_path)
                im_array = np.array(image)
                data.append(im_array)
                class_labels.append(label)

                """
                # For debugging
                plt.figure(1)
                plt.clf()
                plt.imshow(image)
                plt.pause(0.01)
                """

    data = np.array(data)
    data = data.reshape(-1, 3, 64, 64).astype("uint16")
    #print(data.shape)
    class_labels = np.array(class_labels)

    train_data, test_data, train_labels, test_labels = train_test_split(data,
                                                                        class_labels,
                                                                        test_size=0.2,
                                                                        random_state=42)
    """plt.figure()
    plt.imshow(train_data[0])
    plt.show()"""

    """print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)
    print("Training labels shape:", train_labels.shape)
    print("Testing labels shape:", test_labels.shape)"""

    #train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
    #test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    return train_data, test_data, train_labels, test_labels

def create_sequential_model():
    model = tf.keras.models.Sequential()
    # Flattening layer
    model.add(tf.keras.layers.Flatten(input_shape=(3, 64, 64)))
    # 1st layer of 100 fully-connected neurons
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    # 2nd layer of 100 fully-connected neurons
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    # Output layer of 10 neurons
    model.add(tf.keras.layers.Dense(2))

    return model

# Start of main code
data_paths = [(r'C:\Users\oksan\anaconda3\envs\dataml_200\ex2\GTSRB_subset_2\GTSRB_subset_2\class1', 0),
               (r'C:\Users\oksan\anaconda3\envs\dataml_200\ex2\GTSRB_subset_2\GTSRB_subset_2\class2', 1)]

train_data, test_data, train_labels, test_labels = load_data(data_paths)

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print("Training labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)

mlp_model = create_sequential_model()

print(mlp_model.summary())
mlp_model.compile(optimizer='SGD',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
history = mlp_model.fit(train_data,
                        train_labels,
                        epochs=10,
                        batch_size=100)

plt.plot(history.history['loss'])

pred_labels = mlp_model.predict(test_data)
print(pred_labels)
pred_labels_max = np.argmax(pred_labels, axis=1)
print(pred_labels_max)
test_acc = 1 - np.count_nonzero(test_labels-pred_labels_max)/len(test_labels)

print(test_acc)
plt.show()

"""mlp_model = tf.keras.models.Sequential()
# Flattening layer
mlp_model.add(tf.keras.layers.Flatten(input_shape=(3, 64, 64)))
# 1st layer of 100 fully-connected neurons
mlp_model.add(tf.keras.layers.Dense(10, activation="relu"))
# 2nd layer of 100 fully-connected neurons
mlp_model.add(tf.keras.layers.Dense(10, activation="relu"))
# Output layer of 10 neurons
mlp_model.add(tf.keras.layers.Dense(2, activation="softmax"))"""
