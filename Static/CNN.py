import cv2
import os
import numpy as np
import tensorflow as tf
from Static import HEIGHT, WIDTH, PATH_IMG, PATH_IMG_BENIGN

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def load_images_from_folder(folder):
    print("#Loading images...")
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


#######################################################


def pre_process(images_from_disk):
    print("#PREPROCESSING..")
    res_image = []
    dims = (HEIGHT, WIDTH)
    for iterator in range(len(images_from_disk)):
        res = cv2.resize(images_from_disk[iterator], dims, interpolation=cv2.INTER_LINEAR)
        cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)
        res = tf.convert_to_tensor(res, dtype=tf.float32)
        res_image.append(res)

    return res_image


###################################################

def build_model():
    print("##Building model...")
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(HEIGHT, WIDTH, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV - 3
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Output
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train_model(benign, malicious):
    model = build_model()
    print("# Keras: Training model..")
    batch_size = 32
    epochs = 15
    labels = [0 for _ in benign] + [1 for _ in malicious]
    labels = np.array(labels)
    data = benign + malicious
    data = np.array(data)
    print(data.shape)
    history = model.fit(data, labels, batch_size=batch_size, epochs=epochs,
                        validation_split=0.25,
                        shuffle=True)
    plotter(history)
    # model.save_weights()

def plotter(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def driver():
    benign = load_images_from_folder(PATH_IMG_BENIGN)
    mal = load_images_from_folder(PATH_IMG)
    mal = pre_process(mal)
    benign = pre_process(benign)
    build_model()
    train_model(benign, mal)


driver()
