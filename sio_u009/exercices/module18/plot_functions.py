"""This module contains tools to plot data"""
from pandas import DataFrame
from plotnine import aes
from plotnine import geom_abline
from plotnine import geom_point
from plotnine import ggplot
from statsmodels.regression.linear_model import RegressionResultsWrapper

def plot_simple_regression_model(data_frame: DataFrame, model: RegressionResultsWrapper):
    """Plot a simple linear regression model according to it's DataFrame

    Args:
        data_frame (DataFrame): DataFrame from which the regression model was extracted
        model (RegressionResultsWrapper): The model wrapper

    Returns:
        None
    """

    intercept, x_name = model.model.data.xnames
    y_name = model.model.data.ynames

    plot = (
        ggplot(data_frame, aes(x=x_name, y=y_name))
        + geom_point()
        + geom_abline(aes(intercept=model.params.get(intercept), slope=model.params.get(x_name)), color='red')
    )

    print(plot)
