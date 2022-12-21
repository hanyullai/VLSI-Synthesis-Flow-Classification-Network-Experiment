import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D,Flatten,MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix
import itertools
import sys

input_type, model_type, output_type = sys.argv[1], sys.argv[2], sys.argv[3]

def read_csv_data(csv_filename, header=None, num_funcs=6, shuffle_data=True):
    # read data
    data = pd.read_csv(csv_filename, header=header)
    data = data.values
    if (shuffle_data):
        data = shuffle(data)

    # extract X and Y
    X = data[:, :-1]#.reshape((-1, int((data.shape[1]-1)/num_funcs), num_funcs))
    X = data[:, :-1].reshape(X.shape[0], 1, 24, 6)
    Y = data[:, -1]

    # convert Y to one-hot vectors
    Y = to_categorical(Y, num_classes=Y.max()+1)

    return X, Y

from keras.callbacks import Callback
from IPython.display import clear_output
class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1


    def plot_history(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        clear_output(wait=True)
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        plt.savefig('./pic/{}_{}_{}_loss.png'.format(input_type, model_type, output_type), bbox_inches='tight')

plot_learning = PlotLearning()

folder_path = 'datasets/64bitGF/delay/'
csv_filename = 'test.csv'
X, Y = read_csv_data(folder_path + csv_filename, header=None, num_funcs=6)
print("X.shape = ", X.shape)
print("Y.shape = ", Y.shape)

if input_type == 'one-hot':
    pass
elif input_type == 'index':
    X = np.argmax(X, axis=3)
elif input_type == 'embedding':
    X = np.argmax(X, axis=3)
else:
    assert False

if output_type == 'cls':
    pass
elif output_type == 'reg':
    Y = np.argmax(Y, axis=1)
else:
    assert False

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)


if model_type == 'cnn':
    from model.cnn import get_model
    model = get_model(input_type, output_type)
elif model_type == 'mlp':
    from model.mlp import get_model
    model = get_model(input_type, output_type)
elif model_type == 'lstm':
    from model.lstm import get_model
    model = get_model(input_type, output_type)
else:
    assert False

model.summary()

model.fit(X_train, Y_train,
          batch_size=64,
          epochs=100,
          verbose=2,
          validation_data=(X_test, Y_test), callbacks=[plot_learning, EarlyStopping(patience=2)])

import h5py
model.save('dac18_GF_delay_model_save_test.h5')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          print_raw_cm=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if (print_raw_cm):
            print("Normalized confusion matrix")
    else:
        if (print_raw_cm):
            print('Confusion matrix, without normalization')

    if (print_raw_cm):
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate(model, X_test, Y_test, classes, NAME):
    """evaluate dataset and plot confusion matrix"""
    Y_pred = model.predict(X_test)
    if output_type == 'cls':
        Y_pred_label = np.argmax(Y_pred, axis=-1)
    elif output_type == 'reg':
        Y_pred_label = [int((pred[0] + 0.5) // 1) for pred in Y_pred]

    Y_test_label = np.argmax(Y_test, axis=-1)

    print("Prediction accuracy = %f" %(float(sum(Y_pred_label == Y_test_label))/Y_test_label.shape[0]))
    cnf_matrix = confusion_matrix(Y_test_label, Y_pred_label)


    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix\n Angel-class:class-0\n Devil-class:class-6')
    
    plt.savefig('./pic/{}_{}_{}_{}.png'.format(input_type, model_type, output_type, NAME), bbox_inches='tight')



# evaluate(model, X_test, Y_test, classes=range(0,7), NAME="origin")
plot_learning.plot_history()


folder_path = 'datasets/64bitGF/delay/'
csv_filename = 'test2.csv'
X, Y = read_csv_data(folder_path + csv_filename, header=None, num_funcs=6)
if input_type == 'one-hot':
    pass
elif input_type == 'index':
    X = np.argmax(X, axis=3)
elif input_type == 'embedding':
    X = np.argmax(X, axis=3)
else:
    assert False

print(X.shape, Y.shape)
evaluate(model, X, Y, classes=range(0, 7),NAME = "1")


folder_path = 'datasets/64bitGF/delay/'
csv_filename = 'test3.csv'
X, Y = read_csv_data(folder_path + csv_filename, header=None, num_funcs=6)
if input_type == 'one-hot':
    pass
elif input_type == 'index':
    X = np.argmax(X, axis=3)
elif input_type == 'embedding':
    X = np.argmax(X, axis=3)
else:
    assert False

evaluate(model, X, Y, classes=range(0, 7),NAME = "2")


folder_path = 'datasets/64bitGF/delay/'
csv_filename = 'test4.csv'
X, Y = read_csv_data(folder_path + csv_filename, header=None, num_funcs=6)
if input_type == 'one-hot':
    pass
elif input_type == 'index':
    X = np.argmax(X, axis=3)
elif input_type == 'embedding':
    X = np.argmax(X, axis=3)
else:
    assert False

evaluate(model, X, Y, classes=range(0, 7),NAME = "3")


