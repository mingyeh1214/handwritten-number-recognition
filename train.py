import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.math import confusion_matrix
import imutils


def plot_background_mnist(col_num, row_num):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    fig, axs = plt.subplots(row_num, col_num)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(row_num):
        for j in range(col_num):
            idx = i * col_num + j
            print(idx)
            axs[i, j].imshow(x_train[idx].reshape(
                28, 28), cmap='gray', aspect='auto')
            axs[i, j].axis('off')
    fig.savefig("test.png", bbox_inches='tight', pad_inches=0)


def mnist_data():
    img_rows = img_cols = 28
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def image_grey2black(image_origin, threshold):
    image_turn = np.empty_like(image_origin)
    image_turn[image_origin < threshold] = 0
    image_turn[image_origin >= threshold] = 255
    return image_turn


def images_grey2black(images_origin, threshold):
    images_turn = []
    for i in range(images_origin.shape[0]):
        images_turn.append(image_grey2black(images_origin[i], threshold))
    images_turn = np.array(images_turn)
    return images_turn


def image_rotate(image_origin, angle):
    image_turn = imutils.rotate(image_origin, angle)
    return image_turn


def images_rotate(images_origin, angle):
    images_turn = []
    for i in range(images_origin.shape[0]):
        images_turn.append(image_rotate(images_origin[i], angle))
    images_turn = np.array(images_turn)
    return images_turn


def mnist_data_preprocess():
    img_rows = img_cols = 28
    num_classes = 10
    thresholde = 255 / 8
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train_ = y_train
    x_test = images_grey2black(x_test, thresholde)
    x_train_rotate1 = images_rotate(x_train, 20)
    x_train_rotate2 = images_rotate(x_train, -20)
    x_train = np.append(x_train, x_train_rotate1, axis=0)
    y_train = np.append(y_train, y_train_, axis=0)
    x_train = np.append(x_train, x_train_rotate2, axis=0)
    y_train = np.append(y_train, y_train_, axis=0)
    x_train = images_grey2black(x_train, thresholde)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def model_NN(x_train, y_train):
    batch_size = 512
    epochs = 10
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    history = model.fit(x=x_train, y=y_train,
                        batch_size=batch_size, epochs=epochs)
    return model, history


def model_CNN(x_train, y_train, x_test, y_test):
    # mini batch gradient descent ftw
    batch_size = 128
    # very short training time
    epochs = 1
    # build our model
    model = Sequential()
    # convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=(28, 28, 1)))
    # again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    # flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    # fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    # one more dropout for convergence' sake :)
    model.add(Dropout(0.5))
    # output a softmax to squash the matrix into output probabilities
    model.add(Dense(10, activation='softmax'))
    # Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
    # categorical ce since we have multiple classes (10)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])
    # train that ish!
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    return model, history


def model_performance(model, x_test, y_test):
    test_score = model.evaluate(x_test, y_test)
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test = np.argmax(y_test, axis=1)
    confusion_mtx = confusion_matrix(y_test, y_pred)
    return test_score, y_pred, confusion_mtx


def plot_original_xtrain(x_train, y_train):
    y_train = np.argmax(y_train, axis=1)
    fig, axs = plt.subplots(2, 10, figsize=(30, 6))
    fig.tight_layout()
    for i in range(0, 10):
        sample = x_train[i]
        label = y_train[i]
        axs[0, i].imshow(sample, cmap='gray')
        axs[0, i].set_title("Label: {}".format(label), fontsize=14)
        sample = x_train[y_train == i][0]
        label = y_train[y_train == i][0]
        axs[1, i].imshow(sample, cmap='gray')
        axs[1, i].set_title("Label: {}".format(label), fontsize=14)
    return fig


def plot_predict_xtest(x_test, y_test, y_pred):
    y_test = np.argmax(y_test, axis=1)
    fig, axs = plt.subplots(2, 10, figsize=(30, 6))
    fig.tight_layout()
    for i in range(0, 10):
        sample = x_test[i]
        label_true = y_test[i]
        label_pred = y_pred[i]
        axs[0, i].imshow(sample.reshape(28, 28), cmap='gray')
        axs[0, i].set_title("Label: {}, Pred: {}".format(
            label_true, label_pred), fontsize=14)
        sample = x_test[y_test == i][0]
        label_true = y_test[y_test == i][0]
        label_pred = y_pred[y_test == i][0]
        axs[1, i].imshow(sample, cmap='gray')
        axs[1, i].set_title("Label: {}, Pred: {}".format(
            label_true, label_pred), fontsize=14)
    return fig


def plot_accuracy(history, test_acc):
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.title('Model Accuracy(test acc:{})'.format(test_acc))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    return plt


def plot_loss(history, test_loss):
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.title('Model Loss(test loss:{})'.format(test_loss))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    return plt


def plot_confusion_matrix(confusion_mtx):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues",
                     cbar=False, linewidths=.2, linecolor="grey", annot_kws={"fontsize": 16})
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    ax.set_title('Confusion Matrix', fontsize=24)
    return fig


def run_NN():
    x_train, y_train, x_test, y_test = mnist_data()
    model, history = model_NN(x_train, y_train)
    test_score, y_pred, confusion_mtx = model_performance(
        model, x_test, y_test)
    model.save("./models/NN.h5")
    fig = plot_original_xtrain(x_train, y_train)
    fig.savefig("./static/images/NN_original_xtrain.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_accuracy(history, round(test_score[1], 4))
    fig.savefig("./static/images/NN_accuracy.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_loss(history, round(test_score[0], 4))
    fig.savefig("./static/images/NN_loss.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_predict_xtest(x_test, y_test, y_pred)
    fig.savefig("./static/images/NN_predict_xtest.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_confusion_matrix(confusion_mtx)
    fig.savefig("./static/images/NN_confusion_matrix.png",
                bbox_inches='tight', pad_inches=0)


def run_CNN():
    x_train, y_train, x_test, y_test = mnist_data()
    model, history = model_CNN(x_train, y_train, x_test, y_test)
    test_score, y_pred, confusion_mtx = model_performance(
        model, x_test, y_test)
    model.save("./models/CNN.h5")
    fig = plot_original_xtrain(x_train, y_train)
    fig.savefig("./static/images/CNN_original_xtrain.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_accuracy(history, round(test_score[1], 4))
    fig.savefig("./static/images/CNN_accuracy.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_loss(history, round(test_score[0], 4))
    fig.savefig("./static/images/CNN_loss.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_predict_xtest(x_test, y_test, y_pred)
    fig.savefig("./static/images/CNN_predict_xtest.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_confusion_matrix(confusion_mtx)
    fig.savefig("./static/images/CNN_confusion_matrix.png",
                bbox_inches='tight', pad_inches=0)


def run_NN2():
    x_train, y_train, x_test, y_test = mnist_data_preprocess()
    model, history = model_NN(x_train, y_train)
    test_score, y_pred, confusion_mtx = model_performance(
        model, x_test, y_test)
    model.save("./models/NN2.h5")
    fig = plot_original_xtrain(x_train, y_train)
    fig.savefig("./static/images/NN2_original_xtrain.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_accuracy(history, round(test_score[1], 4))
    fig.savefig("./static/images/NN2_accuracy.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_loss(history, round(test_score[0], 4))
    fig.savefig("./static/images/NN2_loss.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_predict_xtest(x_test, y_test, y_pred)
    fig.savefig("./static/images/NN2_predict_xtest.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_confusion_matrix(confusion_mtx)
    fig.savefig("./static/images/NN2_confusion_matrix.png",
                bbox_inches='tight', pad_inches=0)


def run_CNN2():
    x_train, y_train, x_test, y_test = mnist_data_preprocess()
    model, history = model_CNN(x_train, y_train, x_test, y_test)
    test_score, y_pred, confusion_mtx = model_performance(
        model, x_test, y_test)
    model.save("./models/CNN2.h5")
    fig = plot_original_xtrain(x_train, y_train)
    fig.savefig("./static/images/CNN2_original_xtrain.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_accuracy(history, round(test_score[1], 4))
    fig.savefig("./static/images/CNN2_accuracy.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_loss(history, round(test_score[0], 4))
    fig.savefig("./static/images/CNN2_loss.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_predict_xtest(x_test, y_test, y_pred)
    fig.savefig("./static/images/CNN2_predict_xtest.png",
                bbox_inches='tight', pad_inches=0)
    fig = plot_confusion_matrix(confusion_mtx)
    fig.savefig("./static/images/CNN2_confusion_matrix.png",
                bbox_inches='tight', pad_inches=0)


def run_models():
    run_NN()
    run_NN2()
    run_CNN()
    run_CNN2()
