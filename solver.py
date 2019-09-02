from model import create_model
from board import TsumegoBoard
from utils import get_problems, print_from_collection
import numpy as np

WEIGHTS_PATH = 'weights.h5'


def init_model():
    model = create_model()
    model.load_weights(WEIGHTS_PATH)
    return model


class TsumegoNode(TsumegoBoard):
    def find_children(self, model):
        if self.terminal:
            return set()

        possible_moves = self.predict(model)

        return {self.make_move(x, y) for x, y in zip(*possible_moves.nonzero())}

    def find_random_children(self, model):
        pass

    def predict(self, model, threshold=0.02):
        predicted = model.predict([[[self.board]]]).reshape((19, 19))
        predicted[self.legal_moves() == 0] = 0  # abandon illegal moves
        predicted[predicted < threshold] = 0  # abandon unlikely moves
        predicted = predicted / np.sum(predicted)

        return predicted


if __name__ == '__main__':
    model = init_model()

    problems = get_problems()
    print_from_collection(problems, 1)
    p = problems['problem'][1]

    t = TsumegoNode(p, 0, [])
    print(t.predict(model))
