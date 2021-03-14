"""Main"""
import numpy

from sio_u009.algorithm_components.polynomial import Polynomial
from sio_u009.algorithm_components.display import plot_polynomial_over
from sio_u009.algorithm_components.validation_functions import ValidationModel
from sio_u009.algorithm_components.validation_functions import linear_regressive_predictive_model
from sio_u009.algorithm_components.validation_functions import quadratic_loss


def main():
    """Main"""
    numpy.random.seed(42)
    polynomial = Polynomial([0.03, 0.2, -1, -10, 100], 20)

    plot_polynomial_over(polynomial)

    validation_model = ValidationModel(quadratic_loss, linear_regressive_predictive_model, polynomial.data, 25)
    validation_model.plot_objective_function(0.1)

if __name__ == "__main__":
    main()
