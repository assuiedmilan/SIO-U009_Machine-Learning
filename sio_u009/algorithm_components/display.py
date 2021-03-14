"""This module contains plot functions"""

import numpy as np
import matplotlib.pyplot as plt

from sio_u009.algorithm_components.polynomial import Polynomial


def plot_polynomial_over(polynomial: Polynomial):
    """Plot a polynomial of requested points

    Args:
        polynomial (algorithm_components.Polynomial): The polynomial to plot
    """

    _, axes = plt.subplots()

    # Affichage des points
    axes.plot(polynomial.data[0], polynomial.data[1], 'o')

    # Affichage de la fonction échantillonnée
    axes.plot(np.linspace(-10, 10, 100), np.polyval(polynomial.coefficients, np.linspace(-10, 10, 100)), color='black', linewidth=3)

    # Axes & titre
    axes.set_title('Data set')
    axes.set_ylabel('y', fontsize=16)
    axes.set_xlabel('x', fontsize=16)
    plt.show()
