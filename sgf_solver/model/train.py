import os
import h5py
import numpy as np

from sgf_solver.model import create_model
from sgf_solver.constants import PROBLEM_DATASET


def load_problems():
    with h5py.File(PROBLEM_DATASET, 'r') as dataset:
        problems = np.array(dataset['problems'])
        values = np.array(dataset['values'])
        answers = np.array(dataset['answers'])

    problems = problems.reshape((problems.shape[0], 1, 19, 19))
    values = values.reshape((values.shape[0], 1))
    answers = answers.reshape((answers.shape[0], -1))

    return problems, values, answers


def train_model(problems, values, answers):
    model = create_model()
    model.summary()

    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')

    for i in range(5):
        model.fit(problems, [values, answers], batch_size=256, epochs=10, initial_epoch=i*10)
        model.save_weights('weights.h5')


if __name__ == '__main__':
    problems, values, answers = load_problems()

    train_model(problems, values, answers)
