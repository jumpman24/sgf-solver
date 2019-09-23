import os

BOARD_SHAPE = (19, 19)

INPUT_DATA_SHAPE = (2, 19, 19)
CHANNELS_AMOUNT = 16
RESIDUAL_BLOCKS = 4
L2_CONST = 1e-4

base_path = os.path.dirname(__file__)

PROBLEM_PATH = os.path.join(base_path, os.pardir, 'data')
PROBLEM_DATASET = os.path.join(base_path, os.pardir, 'cho_chikun_{}.h5')
WEIGHTS_PATH = os.path.join(base_path, f'model/weights_{CHANNELS_AMOUNT}x{RESIDUAL_BLOCKS}.h5')
