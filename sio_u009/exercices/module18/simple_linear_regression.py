"""This module contains functions for simple linear regression"""

from pandas import DataFrame
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures
from statsmodels.regression.linear_model import RegressionResultsWrapper

from sio_u009.exercices.module18.plot_functions import plot_simple_regression_model

class SimpleLinearRegression:
    """A Simple Linear Regression class for data frames

    Args:
       set_to_adjust (DataFrame): The DataFrame used for this model

    Attributes:
        __set_to_adjust (DataFrame): The DataFrame used for this model
        __model (RegressionResultsWrapper): The model wrapper

    """
    def __init__(self, set_to_adjust: DataFrame):
        self.__set_to_adjust = set_to_adjust
        self.__model = None

    @property
    def model(self) -> RegressionResultsWrapper:
        """The computed model"""
        return self.__model

    def adjust_model_for(self, x_data_name, y_data_name) -> RegressionResultsWrapper:
        """Compute a simple linear regression model for two data frame datas

        Args:
            x_data_name (str): Data to be represented as abscess
            y_data_name (str): Data to be represented as ordinates

        Returns:
            The wrapper containing the model data
        """

        self.__model = smf.ols('{} ~ {}'.format(x_data_name, y_data_name), self.__set_to_adjust).fit()

    def adjust_and_plot_model_for(self, x_data_name, y_data_name) -> RegressionResultsWrapper:
        """Compute a simple linear regression model for two data frame datas and plot it using ggplot

        Args:
            x_data_name (str): Data to be represented as abscess
            y_data_name (str): Data to be represented as ordinates

        Returns:
            The wrapper containing the model data
        """

        self.adjust_model_for(x_data_name, y_data_name)
        plot_simple_regression_model(self.__set_to_adjust, self.__model)



    def get_variability_percentage(self) -> float:
        """Return the variability percentage that explain the y-data by the x-data

        Returns:
            A float
        """

        return round(100*self.__model.rsquared, 1)

    def print_variability_percentage(self):
        """Print the variability percentage that explain the y-data by the x-data

        Returns:
            None
        """

        variability_percentage = self.get_variability_percentage()
        _, x_name = self.__model.model.data.xnames
        y_name = self.__model.model.data.ynames

        print("The percentage of the variability of {} is explained at {}% by {}".format(y_name, variability_percentage, x_name))

    def get_mean_quadratic_error(self) -> float:
        """Return the mean quadratic error expected if we use that model to predict the y-data by the x-data

        Returns:
            A float
        """

        _, x_name = self.__model.model.data.xnames
        set_data_frame = DataFrame(getattr(self.__set_to_adjust, x_name))

        return eval_measures.rmse(getattr(self.__set_to_adjust, x_name), self.__model.predict(set_data_frame)) ** 2
