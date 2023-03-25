import tensorflow as tf
import numpy as np
import os
import PIL
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    #train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)

    # Normalize our data
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

def get_test_accuracy(test_labels, pred_labels):

    """correct_pred = 0
    test_total = 0
    print(test_labels)
    print(pred_labels)
    for i in range(len(test_labels)):
        test_total += 1
        if pred_labels[i] == test_labels[i]:
            correct_pred += 1

    accuracy = correct_pred/test_total"""
    # Use ready made function from scikit learning to measure the accuracy of the model
    accuracy = accuracy_score(pred_labels, test_labels)

    return accuracy

# Start of main code
data_paths = [(r'C:\Users\oksan\anaconda3\envs\tf\ex2_dataml200\GTSRB_subset_2\GTSRB_subset_2\class1', 0),
               (r'C:\Users\oksan\anaconda3\envs\tf\ex2_dataml200\GTSRB_subset_2\GTSRB_subset_2\class2', 1)]

train_data, test_data, train_labels, test_labels = load_data(data_paths)

"""
# Prints for debugging
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print("Training labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)
"""

mlp_model = create_sequential_model()

print(mlp_model.summary())

# If the output of the model is normalized (output layer using softmax function for example),
# we can set the from_logits=False
mlp_model.compile(optimizer='SGD',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
history = mlp_model.fit(train_data,
                        train_labels,
                        epochs=10,
                        batch_size=10)

plt.plot(history.history['loss'])

pred_labels = mlp_model.predict(test_data)
pred_labels_max = np.argmax(pred_labels, axis=1)

test_acc = get_test_accuracy(test_labels, pred_labels_max)

print()
print(f"Accuracy for test set: {test_acc:.02f}")
plt.show()
