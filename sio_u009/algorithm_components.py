"""Define components for algorithms"""

from typing import List
from typing import Tuple

import numpy as np


class Polynomial:
    """This class defines a polynomial and can generate noisy data points

    Args:
        coefficients (List[float]): Coefficients of the polynomial
        number_of_points (int): Number of points to generate
    """

    def __init__(self, coefficients: List[float], number_of_points):
        self.__coefficients = coefficients
        self.__data = None

        self.__compute_data(number_of_points)

    @property
    def coefficients(self) -> List[float]:
        """Returns the polynomial coefficients

        Returns:
            a List[float] of coefficients
        """

        return self.__coefficients

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the polynomial values

        Returns:
            a Tuple[np.ndarray, np.ndarray] of values [x, y] of y = P(x) + noise
        """

        return self.__data

    def __compute_data(self, number_of_points: int):
        """Generate a number of points randomly over the polynomial function

        Args:
            number_of_points (int): Number of points to generate

        Returns
            A tuple of numpy array associating the random points with their polynomial value + a random noise value
        """

        x = np.random.uniform(-10, 10, number_of_points)
        y = np.polyval(self.coefficients, x) + np.random.normal(0.0, 15.0, number_of_points)
        self.__data = x.reshape(-1, 1), y
