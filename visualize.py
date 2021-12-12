from matplotlib import pyplot as plt


def show_loss(history, no_of_epochs, filename="", show=True):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 5])
    plt.legend(loc='upper left')
    if show: plt.show()
    plt.savefig(filename)
    plt.figure().clear()


def show_acc(history, no_of_epochs, filename="", show=True):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 1])
    plt.legend(loc='upper left')
    if show: plt.show()
    plt.savefig(filename)
    plt.figure().clear()


def show_evaluation(history, no_of_epochs, filename, show=True):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([-0.05, 5])
    plt.legend(loc='upper left')
    plt.savefig(filename)
    if show: plt.show()
    plt.figure().clear()

