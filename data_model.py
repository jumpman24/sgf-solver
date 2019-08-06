import numpy as np


class Problem:
    def __init__(self,
                 black_planes, black_liberties_planes,
                 white_planes, white_liberty_planes,
                 black_to_play, white_to_play):
        assert black_plane.shape == [1, 19, 19]
        assert black_liberties_planes.shape == [1, 19, 19]
        assert white_plane.shape == [1, 19, 19]
        assert white_liberty_planes.shape == [1, 19, 19]
        assert black_to_play.shape == [1, 19, 19]
        assert white_to_play.shape == [1, 19, 19]

        self.data = np.append([
            black_planes, black_liberties_planes,
            white_planes, white_liberty_planes,
            black_to_play, white_to_play], axis=0)

