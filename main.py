from tensorflow.keras import Input, optimizers, losses, metrics, utils, callbacks
import os
from errno import EEXIST
import argparse
from models import get_model
from data_loader import data_loader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--idx",
                    help='CNN: 0, RNN: 1, DNN:2',
                    default=0,
                    type=int)
parser.add_argument("--restore",
                    help='0:False, 1:True',
                    default=0,
                    type=int)


def mkdir_p(my_path):
    try:
        os.makedirs(my_path)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(my_path):
            pass
        else:
            raise


class SaveBestModel(callbacks.Callback):
    def __init__(self, save_best_metric='val_categorical_accuracy', this_max=True):
        super().__init__()
        self.save_best_metric = save_best_metric
        self.best_epoch = 0
        self.max = this_max
        self.best_weights = 0
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                print(f'\nEpoch {epoch}: {self.save_best_metric} improved '
                      f'from {self.best:.5f} to {metric_value:.5f}, ')
                self.best = metric_value
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
            else:
                print(f'\nEpoch {epoch}: '
                      f'{self.save_best_metric} did not improve from {self.best:.5f}')
        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


if __name__ == '__main__':
    args = parser.parse_args()

    model = get_model(args.idx)

    if args.idx == 0:
        filename = "CNN\\"
    elif args.idx == 1:
        filename = "RNN\\"
    else:
        filename = "DNN\\"

    mkdir_p(filename)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, label = data_loader()

    if args.idx == 0:
        model.build(input_shape=(None, 11, 3, 1))
        model.call(Input(shape=(11, 3, 1)))
    elif args.idx == 1:
        model.build(input_shape=(None, 11, 3))
        model.call(Input(shape=(11, 3)))
    else:
        model.build(input_shape=(None, 33))
        model.call(Input(shape=33))

    utils.plot_model(
        model.build_graph(), to_file=filename + 'model.png',
        show_shapes=True,
        show_layer_names=True, rankdir='TB', expand_nested=True,
    )

    model.summary()

    model.compile(optimizer=optimizers.Adam(10e-3),
                  loss=losses.CategoricalCrossentropy(),
                  metrics=[metrics.CategoricalAccuracy()],
                  )

    save_best_model = SaveBestModel()
    callback = [callbacks.EarlyStopping(patience=10,
                                        monitor='val_categorical_accuracy'),
                save_best_model]

    if args.idx == 0:
        X_train = X_train.reshape(X_train.shape[0], 11, 3, 1)
        X_val = X_val.reshape(X_val.shape[0], 11, 3, 1)
        X_test = X_test.reshape(X_test.shape[0], 11, 3, 1)
    elif args.idx == 1:
        X_train = X_train.reshape(X_train.shape[0], 11, 3)
        X_val = X_val.reshape(X_val.shape[0], 11, 3)
        X_test = X_test.reshape(X_test.shape[0], 11, 3)

    if args.restore:
        model.load_weights(filename + "weights.best.hdf5")
    else:
        history = model.fit(X_train, Y_train,
                            batch_size=1,
                            epochs=200,
                            validation_data=(X_val, Y_val),
                            callbacks=callback
                            )

        f = open(filename + "best_epoch.txt", "w")
        f.write(f"best epoch :{save_best_model.best_epoch}")
        f.close()

        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.axvline(x=callback[1].best_epoch)
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(filename + "Accuracy plot")
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.axvline(x=callback[1].best_epoch)
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(filename + "Loss plot")
        plt.show()

        model.set_weights(save_best_model.best_weights)
        model.save_weights(filename + "weights.best.hdf5")

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(Y_test, axis=1)

    m = metrics.Accuracy()
    m.update_state(y_pred, y_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    ax = sns.heatmap(conf_mat, annot=True, cmap='Blues', xticklabels=label, yticklabels=label)
    ax.invert_yaxis()
    plt.title(f"Test data Confusion matrix at epoch:{save_best_model.best_epoch} "
              f"(Accuracy:{m.result().numpy() * 100:.2f}%)")
    plt.show()
