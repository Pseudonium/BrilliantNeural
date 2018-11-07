import numpy as np
import collections
import itertools


class Perceptron:
    Classifier = collections.namedtuple('Classifier', ['weight', 'bias'])

    def __init__(self, data):
        self.points = list(
            (Perceptron.point_to_vector(xi), yi)
            for xi, yi in data
        )
        self.shape = self.points[0][0].shape

    @staticmethod
    def point_to_vector(datum):
        return np.matrix(
            list(
                [coord] for coord in datum
            )
        )

    @property
    def classifier(self):
        bias = 0
        count = 0
        weight = np.zeros(self.shape)
        for xi, yi in itertools.cycle(self.points):
            if count == 1000:
                print("Maximum time exceeded.")
                break
            num_misclassified = sum(
                1 if yi * (weight.transpose() * xi + bias) <= 0
                else 0 for xi, yi in self.points
            )
            if num_misclassified == 0:
                print("All points classified.")
                break
            is_misclassified = yi * (weight.transpose() * xi + bias)
            if is_misclassified <= 0:
                weight += yi * xi
                bias += yi
                print("New weight: ", weight)
                print("Bias: ", bias)
            count += 1
        return Perceptron.Classifier(weight, bias)


if __name__ == "__main__":
    x_points = [
        [
            [-1, 1],
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
    y = Perceptron(x_points)
    print(y.classifier)
    # print(list(y.points))
