import numpy as np

from data_parser import get_features, get_lables

feature_array = get_features()
labels_array = get_lables()

train_data = feature_array[:700]
train_labels = labels_array[:700]
test_data = feature_array[700:]
test_labels = labels_array[700:]

theta = np.zeros((361, 1))


def sigmoid(z):
    return np.divide(1, np.add(1, np.exp(np.negative(z))))


def cost_function(theta, x, y):
    m = len(x)

    sig_pos = sigmoid(x.nonzero())
    sig_neg = sigmoid(x.zero())
    print(sig_pos)
    print(sig_neg)

print(cost_function(theta, train_data, train_labels))