import cv2
import os
import numpy as np
from Static import HEIGHT, WIDTH, PATH_IMG, PATH_IMG_BENIGN


def load_images_from_folder(folder):
    print("#Loading images...")
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def reshape(images):
    print("##reshaping..")
    img_list = np.zeros(shape=(len(images), HEIGHT, WIDTH, 1), dtype=np.uint8)
    for j in range(len(images)):
        img_list[j, :, :, 0] = np.reshape(list(images[j]), (HEIGHT, WIDTH))

    img_list = img_list.astype('float32')
    img_list /= 255
    return img_list


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def build_model(images):
    reshaped_list = reshape(images)
    print("##Building model...")
    model = Sequential()
    # Conv2D Layers
    model.add(Conv2D(12, (25, 25), padding='same', input_shape=reshaped_list.shape[1:], activation='relu'))
    model.add(Conv2D(12, (25, 25), activation='relu'))
    # Max Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Conv2D Layer
    model.add(Conv2D(12, (13, 13), padding='same', activation='relu'))
    model.add(Conv2D(12, (13, 13), activation='relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flattening Layer
    model.add(Flatten())
    # Dense Layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model

def train_model(benign, malicious):
    model = build_model(malicious)
    print("# Keras: Training model..")
    batch_size = 512
    epochs = 100
    labels = [0 for _ in benign] + [1 for _ in malicious]
    model.fit(benign + malicious, labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.25,
              shuffle=True)
    #model.save_weights()

def driver():
    benign=load_images_from_folder(PATH_IMG_BENIGN)
    mal=load_images_from_folder(PATH_IMG)
    train_model(benign, mal)

driver()