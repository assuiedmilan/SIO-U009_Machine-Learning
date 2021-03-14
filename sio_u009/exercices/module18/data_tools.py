"""This module contains some functions used to manipulate DataFrames and Series"""

from typing import List
from typing import Tuple

from pandas import DataFrame

from sio_u009.exercices.module18.datas import __airports
from sio_u009.exercices.module18.datas import __flights


def get_all_origin_airports_for_flights() -> DataFrame:
    """List of airports from which the flights departed

    Notes:
          The provided flight files contains flights that departed only from New York Airports, so we should get 3 airports from this list.
    Returns:
        A DataFrame containing all the airports which a flight departed from
    """
    return __airports[__airports.faa.isin(__flights.origin)]


def get_all_flights_from_in_the_last_months(origin_faa_code: str, number_of_months: int) -> DataFrame:
    """Get all the flights that departed from the requested airport for the last requested number of months

    Args:
        origin_faa_code (str): airport code
        number_of_months (int): number of months to fetch the data

    Returns:
        A DataFrame
    """

    return __flights[(__flights.dest == origin_faa_code) & (__flights.month == number_of_months)]


def join_new_data_using_common_data(join_to: DataFrame, join_from: DataFrame, common_data: List[str], data_to_join: List[str]) -> DataFrame:
    """Uses common data between two DataFrames to extract information and merge it into one of the DataFrame

    Args:
        join_to (DataFrame): The DataFrame that will receive the data to merge
        join_from (DataFrame): The DataFrame which data will be extracted from
        data_to_join (List[str]): List of the columns to extract from the 'from' DataFrame
        common_data (List[str]): List of the columns in common on which filter the data from the 'from' DataFrame

    Returns:
        An updated DataFrame
    """

    filter_list = common_data + data_to_join
    return join_to.join(join_from.set_index(common_data).filter(items=filter_list), on=common_data)


def extract_test_and_training_set(extract_from: DataFrame, percent_of_set_for_testing: int) -> Tuple[DataFrame, DataFrame]:
    """From a data set, extract a training set and a test set

    Args:
        extract_from (DataFrame): DataFrame containing the datas
        percent_of_set_for_testing (int): Percentage of the DataFrame that will be used for testing

    Notes:
        For reproducibility, a fixed seed is used to generate the extraction. There's no randomness in the process.

    Returns:
        A tuple of DataFrames (test_set, training_set)
    """

    copy_set = extract_from.copy()
    test_set = copy_set.sample(frac=percent_of_set_for_testing / 100, random_state=75)
    training_set = copy_set.drop(test_set.index)

    return test_set, training_set
