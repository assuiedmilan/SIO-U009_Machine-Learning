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

    @property
    def x(self) -> np.ndarray:
        """Returns x values of the polynomial"""

        return self.data[0]

    @property
    def y(self) -> np.ndarray:
        """Returns y values of the polynomial"""

        return self.data[1]

    def value(self, x: [float, np.ndarray, np.poly1d]) -> np.ndarray:
        """Returns the value at x

        Args:
            x (float, np.ndarray, np.poly1d): value over which to compute the polynomial values

        Returns:
            An array containing the values
        """
        return np.polyval(self.coefficients, x)

    def __compute_data(self, number_of_points: int):
        """Generate a number of points randomly over the polynomial function

        Args:
            number_of_points (int): Number of points to generate

        Returns
            A tuple of numpy array associating the random points with their polynomial value + a random noise value
        """

        x = np.random.uniform(-10, 10, number_of_points)
        y = self.value(x) + np.random.normal(0.0, 15.0, number_of_points)
        self.__data = x.reshape(-1, 1), y
