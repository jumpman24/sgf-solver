import os
import h5py
import numpy as np

from sgf_solver.model import create_model
from sgf_solver.constants import PROBLEM_DATASET, WEIGHTS_PATH, INPUT_DATA_SHAPE


def load_problems():
    with h5py.File(PROBLEM_DATASET.format('big'), 'r') as dataset:
        problems = np.array(dataset['problems'], dtype=int)
        values = np.array(dataset['values'], dtype=int)
        answers = np.array(dataset['answers'], dtype=int)

    np.random.seed(0)
    count = problems.shape[0]
    indices = np.arange(count)
    np.random.shuffle(indices)
    problems = problems.reshape((count, *INPUT_DATA_SHAPE))[indices]
    values = values.reshape((count, 1))[indices]
    answers = answers.reshape((count, 361))[indices]

    return problems, values, answers


def train_model(problems, values, answers):
    model = create_model()

    if os.path.exists(WEIGHTS_PATH):
        print("Loading weights")
        model.load_weights(WEIGHTS_PATH)

    model.fit(problems, [values, answers],
              epochs=1,
              batch_size=256,
              validation_split=0.2)
    model.save_weights(WEIGHTS_PATH)


if __name__ == '__main__':
    problems, values, answers = load_problems()
    train_model(problems, values, answers)
