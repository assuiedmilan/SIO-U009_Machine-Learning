# pylint: skip-file
# This module is copied-pasted from the online course

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error


from sio_u009.algorithm_components.polynomial import Polynomial

phi = {
    "x_polynomial": lambda x: np.polyval([1, 2, -4, 4], x),
    "x_abs": lambda x: np.abs(x),
    "x_positive": lambda x: x > 0.0,
    "x_cosine": lambda x: np.cos(x),
    "x_sine": lambda x: np.sin(x)
}

def feature_space_projection(X, phi_functions):
    X_features = [np.apply_along_axis(space_projection_function, 0, X) for space_projection_function in phi_functions]
    return np.concatenate(X_features, axis=1)


def plot_non_regularized_trained_model(learning_polynomial, test_polynomial):
    plot_trained_model(learning_polynomial, test_polynomial, linear_model.LinearRegression())


def plot_regularized_trained_model(learning_polynomial, test_polynomial, regularization_ratio, norms_ratio):
    reg = linear_model.ElasticNet(alpha=regularization_ratio,
                                   copy_X=True,
                                   fit_intercept=True,
                                   l1_ratio=norms_ratio,
                                   max_iter=10000,
                                   normalize=False,
                                   positive=False,
                                   precompute=False,
                                   random_state=None,
                                   selection='random',
                                   tol=0.0001,
                                   warm_start=False
                                   )

    plot_trained_model(learning_polynomial, test_polynomial, reg)

def plot_trained_model(learning_polynomial, test_polynomial, regresion_function):
    X_augmented = feature_space_projection(learning_polynomial.x, phi.values())

    regresion_function.fit(X_augmented, learning_polynomial.y)
    print('\'' + pd.DataFrame({'names': phi.keys(), 'coefs': regresion_function.coef_}).to_string(index=False)[1:])

    y_pred = regresion_function.predict(X_augmented)
    training_error = mean_squared_error(learning_polynomial.y, y_pred)
    print("L'erreur d'entraînement du modèle appris est : %5.2f" % training_error)

    X_test_augmented = feature_space_projection(test_polynomial.x, phi.values())
    y_test_pred = regresion_function.predict(X_test_augmented)
    test_error = mean_squared_error(test_polynomial.y, y_test_pred)
    print("Vrai risque du modèle appris est : %5.2f" % test_error)

    fig, ax = plt.subplots()
    ax.plot(learning_polynomial.x, learning_polynomial.y, 'o')
    ax.set_title('Polynôme')
    ax.set_ylabel('y')
    ax.set_xlabel('x')

    linspace_x = np.linspace(-10, 10, num=100000)
    linspace_x = np.expand_dims(linspace_x, axis=1)

    linspace_X_augmented = feature_space_projection(linspace_x, phi.values())

    y_pred = regresion_function.predict(linspace_X_augmented)
    ax.plot(linspace_x, y_pred, color='red', linewidth=3)

    plt.show()


def main():
    """Main"""
    np.random.seed(42)
    coeffs = [0.03, 0.2, -1, -10, 100]
    learning_polynomial = Polynomial(coeffs, 20)
    test_polynomial = Polynomial(coeffs, 10000000)


    plot_non_regularized_trained_model(learning_polynomial, test_polynomial)
    plot_regularized_trained_model(learning_polynomial, test_polynomial, 0.1, 0.5)



if __name__ == "__main__":
    main()
