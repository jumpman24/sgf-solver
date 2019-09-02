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
    def find_children(self):
        if self.terminal:
            return set()

        return {self.make_move(x, y)
                for x in range(self.board_size)
                for y in range(self.board_size) if self.legal_moves()[x, y] == 1}

    def find_random_children(self, model):
        pass

    def predict(self, model):
        predicted = model.predict([[[self.board]]]).reshape((19, 19))
        predicted[self.legal_moves() == 0] = 0
        predicted[predicted < 0.02] = 0
        predicted = predicted / np.sum(predicted)

        return predicted


if __name__ == '__main__':
    model = init_model()

    problems = get_problems()
    print_from_collection(problems, 1)
    p = problems['problem'][1]

    t = TsumegoNode(p, 0, [])
    print(t.predict(model))
