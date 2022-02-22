import tensorflow
import tensorflow.keras.metrics as metrics
from CNN import load_images_from_folder, pre_process
from ANNmodel import ResidualAttentionNetwork
from Static import PATH_IMG_BENIGN, PATH_IMG
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizer_v2.adam import Adam
from keras.losses import binary_crossentropy
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as k


def runner():
    load_1 = load_images_from_folder(PATH_IMG_BENIGN)
    load_2 = load_images_from_folder(PATH_IMG)
    benign = pre_process(load_1, height=64, width=64)
    malicious = pre_process(load_2, height=64, width=64)
    ## data
    data = benign + malicious
    data = np.array(data)

    ## labels
    labels = [0 for _ in benign] + [1 for _ in malicious]
    labels = np.array(labels)

    model = ResidualAttentionNetwork((64, 64, 3), 1, 'sigmoid').build_model()

    opt = SGD(learning_rate=0.01)
    model.compile(optimizer=opt,
                  loss=binary_crossentropy,
                  metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()])

    model.load_weights('ann_best.h5')

    print(">> weight loaded")

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2, stratify=labels)

    """model.fit(X_train, y_train, batch_size=30, epochs=10,
              validation_split=0.15,
              shuffle=True)
              """

    print(">> evaluating..")
    loss, acc, f1_score, precision, recall = model.evaluate(X_test, y_test)

    print(f1_score)
    print(precision)
    print(recall)

    # pred = model.predict(X_test)
    #    pred = pred.argmax(axis=-1)
    # cf_matrix = confusion_matrix(y_test, pred)
    # sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
    #       fmt='.2%', cmap='Blues')
    # plt.show()


if __name__ == '__main__':
    runner()
