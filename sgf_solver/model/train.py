import os
import h5py
import numpy as np

from sgf_solver.model import create_model
from sgf_solver.constants import PROBLEM_DATASET, WEIGHTS_PATH, INPUT_DATA_SHAPE


def load_problems():
    with h5py.File(PROBLEM_DATASET, 'r') as dataset:
        problems = np.array(dataset['problems'])
        values = np.array(dataset['values'])
        answers = np.array(dataset['answers'])

    problems = problems.reshape((-1, *INPUT_DATA_SHAPE))
    values = values.reshape((-1, 1))
    answers = answers.reshape((-1, 361))

    return problems, values, answers


def train_model(problems, values, answers):
    model = create_model()

    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)

    for i in range(10):
        model.fit(problems, [values, answers],
                  batch_size=128,
                  epochs=50,
                  shuffle=True)
        model.save_weights(WEIGHTS_PATH)


if __name__ == '__main__':
    problems, values, answers = load_problems()

    train_model(problems, values, answers)
