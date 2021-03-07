"""This module contains usable functions for loss and predictive model, as long as a model validation class"""
from typing import Callable
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def quadratic_loss(y1: float, y2:float) -> float:
    """Compute the quadratic loss

    Args:
        y1 (float): first value
        y2 (float): second value

    Returns:
        The quadratic difference between the inputs
    """

    return np.abs(y1 - y2)**2

def absolute_loss(y1: float, y2:float) -> float:
    """Compute the absolute loss

    Args:
        y1 (float): first value
        y2 (float): second value

    Returns:
        The absolute difference between the inputs
    """

    return np.abs(y1 - y2)

def linear_regressive_predictive_model(x: float, theta: np.ndarray) -> float:
    """Predictive model using a linear regression

    Args:
        x (float): value
        theta (np.ndarray): Coefficients of the second order polynomial

    Returns:
        The output of the x value feed to the theta polynomial
    """

    return theta[0] * x + theta[1]

def quadratic_predictive_model(x: float, theta: np.ndarray) -> float:
    """Predictive model using a quadratic regression

    Args:
        x (float): value
        theta (np.ndarray): Coefficients of the third order polynomial

    Returns:
        The output of the x value feed to the theta polynomial
    """

    return theta[0] * x**2 + theta[1] * x + theta[2]


class ValidationModel:
    """Model validation class

    Args:
        loss_function (Callable[[float, float], float]): Loss function
        predictive_model (Callable[[float, Tuple[float]], float]): Predictive model function
        data (Tuple[np.ndarray, np.ndarray]): values [x, y] of y = P(x) + noise
        number_of_points (int): number of points over which to run the validation
    """

    def __init__(self, loss_function: Callable[[float, float], float], predictive_model: Callable[[float, np.ndarray], float], data: Tuple[np.ndarray, np.ndarray], number_of_points: int):
        self.__loss_function = loss_function
        self.__predictive_model = predictive_model
        self.__x = data[0]
        self.__y = data[1]
        self.__number_of_points = number_of_points
        self.__linear_space = np.linspace(-200, 200, self.__number_of_points)

    def empiric_risk(self, theta: np.ndarray) -> float:
        """Return the empiric risk computed over theta value

        Args:
            theta (np.ndarray):

        Returns:
            Average empiric risk
        """

        loss = 0

        for i in range(0,np.max(self.__y.shape)):
            y_hat = self.__predictive_model(self.__x[i], theta)
            loss = loss + self.__loss_function(self.__y[i], y_hat)

        return loss/float(np.max(self.__y.shape))

    @staticmethod
    def regularize(theta: np.ndarray) -> np.float64:
        """Regularization function

        Args:
            theta (np.ndarray):

        Returns:
            The regularized value
        """

        return np.sum( np.abs(theta) )


    def objective(self, r: float, theta: np.ndarray) -> np.ndarray:
        """Objective function

        Args:
            r (float):
            theta (np.ndarray):

        Returns:
            An array which represents the objective value
        """

        return self.empiric_risk(theta) + r * self.regularize(theta)

    def compute_objectives(self, r: float) -> np.ndarray:
        """Compute all objectives results

        Args:
            r (float): tolerance

        Returns:
            An array which contains the objective values
        """

        r_values = np.zeros((self.__number_of_points, self.__number_of_points))

        for i in range(0, self.__number_of_points):
            for j in range(0, self.__number_of_points):
                r_values[i, j] = np.log(self.objective(r, self.compute_theta(i, j)))

        return r_values

    def compute_theta(self, first_index: int, second_index: int) -> np.ndarray:
        """Return theta value computed over a linear space

        Args:
            first_index (int)
            second_index (int)

        Returns:
            A two dimensional array
        """

        return np.array([self.__linear_space[first_index], self.__linear_space[second_index]])

    def plot_objective_function(self, r):
        """Plot objective function

        Args:
            r (int)

        Returns:
            None
        """

        plt.figure()
        axes = plt.axes(projection='3d')
        mesh_a, mesh_b = np.meshgrid(self.__linear_space, self.__linear_space)
        axes.plot_surface(mesh_a, mesh_b, self.compute_objectives(r))
        plt.show()
