import numpy as np


def one_hot(index, size):
    array = np.zeros(size)
    array[index] = 1
    return array


def binary_confusion_matrix(y_true_class, y_pred_class):
    return [[
        np.sum((y_true_class == 1) & (y_pred_class == 1)),
        np.sum((y_true_class == 1) & (y_pred_class == 0)),
    ], [
        np.sum((y_true_class == 0) & (y_pred_class == 1)),
        np.sum((y_true_class == 0) & (y_pred_class == 0)),
    ]]


def confusion_matrix(y_true_class, y_pred_class):
    n = int(max(y_true_class.max(), y_pred_class.max())) + 1
    return [
        [
            np.sum((y_true_class == true) & (y_pred_class == pred))
            for pred in range(n)
        ]
        for true in range(n)
    ]
