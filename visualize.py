from matplotlib import pyplot as plt


def show_loss(history, no_of_epochs):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 10])
    plt.legend(loc='upper left')
    plt.show()


def show_acc(history, no_of_epochs):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 1])
    plt.legend(loc='upper left')
    plt.show()