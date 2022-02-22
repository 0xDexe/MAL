import cv2
import os
import numpy as np
import tensorflow as tf
from Static import HEIGHT, WIDTH, PATH_IMG, PATH_IMG_BENIGN

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizer_v2 import rmsprop
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_images_from_folder(folder):
    print("> loading images...")
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


#######################################################


def pre_process(images_from_disk, height, width):
    print("> preprocessing...")
    res_image = []
    dims = (height, width)
    for iterator in range(len(images_from_disk)):
        res = cv2.resize(images_from_disk[iterator], dims, interpolation=cv2.INTER_LINEAR)
        cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)
        res = tf.convert_to_tensor(res, dtype=tf.float32)
        res_image.append(res)

    return res_image


###################################################

def build_model():
    print(">> Building CNN...")
    model = Sequential()
    model.add(Conv2D(32, (1, 1), padding='same', input_shape=(HEIGHT, WIDTH, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(1, activation='sigmoid'))

    opt = rmsprop.RMSprop(learning_rate=0.1)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC() ])

    return model



def train_model(benign, malicious):
    model = build_model()
    print(">>>training model..")
    batch_size = 50
    epochs = 6
    labels = [0 for _ in benign] + [1 for _ in malicious]
    labels = np.array(labels)

    data = benign + malicious
    data = np.array(data)

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.25,
                                                        random_state=80)
    model.load_weights("cnn_new.h5")
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
     #         validation_split=0.25,
      #        shuffle=True)

    print(">>> evaluating...")
    loss, acc, f1_score, precision, recall = model.evaluate(X_test, y_test)
    print(f1_score)
    print(precision)
    print(recall)

    #pred = model.predict(X_test)

    #model.save_weights("cnn_new.h5")
    print(">>> weight saved ")
    #pred = pred.argmax(axis=-1)
    #cf_matrix = confusion_matrix(y_test, pred)
    #print(cf_matrix)
    #sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
               # fmt='.2%', cmap='Blues')
    #plt.show()


def plotter(history):
    print(history.history.keys())
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


if __name__ == '__main__':
    benign = load_images_from_folder(PATH_IMG_BENIGN)
    mal = load_images_from_folder(PATH_IMG)
    mal = pre_process(mal, HEIGHT, WIDTH)
    benign = pre_process(benign, HEIGHT, WIDTH)
    train_model(benign, mal)

