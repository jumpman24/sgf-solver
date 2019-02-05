import numpy as np

PATH_TO_FEATURES = 'features.txt'
PATH_TO_LABELS = 'labels.txt'


def get_features():
    with open(PATH_TO_FEATURES) as f:
        input_data = f.readlines()
        all_features = []
        for line in input_data:
            features = []
            for item in list(line.strip()):
                features.append(int(item))
            all_features.append(features)
    feature_array = np.array(all_features)
    return feature_array


def get_lables():
    with open(PATH_TO_LABELS) as f:
        input_data = f.readlines()
        all_labels = []
        for line in input_data:
            features = []
            for item in list(line.strip()):
                features.append(float(item))
            all_labels.append(features)
    labels_array = np.array(all_labels)
    return labels_array
