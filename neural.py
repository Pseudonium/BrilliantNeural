import numpy as np


class Perceptron:
    def __init__(self, data):
        self.points = (
            (Perceptron.point_to_vector(xi), yi)
            for xi, yi in data
        )

    @staticmethod
    def point_to_vector(datum):
        return np.matrix(
            list(
                [coord] for coord in datum
            )
        )


if __name__ == "__main__":
    x_points = [
        [
            np.matrix([[-1], [1]]),
            1
        ],
        [
            [0, -1],
            -1
        ],
        [
            [10, 1],
            1
        ]
    ]
    print(
        list(
            (np.matrix(xi), yi)
            for xi, yi in x_points
        )
    )
    z = Perceptron.point_to_vector([-1, 1])
