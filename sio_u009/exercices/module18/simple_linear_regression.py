"""This module contains functions for simple linear regression"""

from pandas import DataFrame
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures
from statsmodels.regression.linear_model import RegressionResultsWrapper

from sio_u009.exercices.module18.plot_functions import plot_simple_regression_model


def adjust_model_for(set_to_adjust: DataFrame, x_data_name, y_data_name) -> RegressionResultsWrapper:
    """Compute a simple linear regression model for two data frame datas

    Args:
        set_to_adjust (DataFrame): The DataFrame
        x_data_name (str): Data to be represented as abscess
        y_data_name (str): Data to be represented as ordinates

    Returns:
        The wrapper containing the model data
    """

    return smf.ols('{} ~ {}'.format(x_data_name, y_data_name), set_to_adjust).fit()

def adjust_and_plot_model_for(set_to_adjust: DataFrame, x_data_name, y_data_name) -> RegressionResultsWrapper:
    """Compute a simple linear regression model for two data frame datas and plot it using ggplot

    Args:
        set_to_adjust (DataFrame): The DataFrame
        x_data_name (str): Data to be represented as abscess
        y_data_name (str): Data to be represented as ordinates

    Returns:
        The wrapper containing the model data
    """

    model = adjust_model_for(set_to_adjust, x_data_name, y_data_name)
    plot_simple_regression_model(set_to_adjust, model)

    return model


def get_variability_percentage(model: RegressionResultsWrapper) -> float:
    """Return the variability percentage that explain the y-data by the x-data

    Args:
        model (RegressionResultsWrapper): The model wrapper

    Returns:
        A float
    """

    return round(100*model.rsquared, 1)

def print_variability_percentage(model: RegressionResultsWrapper):
    """Print the variability percentage that explain the y-data by the x-data

    Args:
        model (RegressionResultsWrapper): The model wrapper

    Returns:
        None
    """

    variability_percentage = get_variability_percentage(model)
    _, x_name = model.model.data.xnames
    y_name = model.model.data.ynames

    print("The percentage of the variability of {} is explained at {}% by {}".format(y_name, variability_percentage, x_name))

def get_mean_quadratic_error(data_frame: DataFrame, model: RegressionResultsWrapper) -> float:
    """Return the mean quadratic error expected if we use that model to predict the y-data by the x-data

    Args:
        data_frame (DataFrame): The DataFrame used for this model
        model (RegressionResultsWrapper): The model wrapper

    Returns:
        A float
    """

    _, x_name = model.model.data.xnames
    set_data_frame = DataFrame(getattr(data_frame, x_name))

    return eval_measures.rmse(getattr(data_frame, x_name), model.predict(set_data_frame)) ** 2
