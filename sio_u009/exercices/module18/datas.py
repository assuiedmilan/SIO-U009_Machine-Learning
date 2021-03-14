"""This module contains the datas used for this learning model"""

import os
from typing import List
from typing import Union

import pandas as pd
import wget
from pandas import DataFrame


def __read(data_file: str, parse_dates: Union[List[str], bool] = False, index_col: int=None):

    data_files_ext = '.csv'
    data_file_with_ext = data_file + data_files_ext
    data_files_folder = os.path.join('./', 'resources')
    data_file_path = os.path.join(data_files_folder, data_file_with_ext)

    if not os.path.isdir(data_files_folder):
        os.mkdir(data_files_folder)

    if not os.path.isfile(data_file_path):
        wget.download('https://raw.githubusercontent.com/iid-ulaval/EEAA-datasets/master/{}'.format(data_file_with_ext), data_file_path)

    return pd.read_csv(data_file_path, parse_dates=parse_dates, index_col=index_col)

__airports = __read('airports')
__flights = __read("flights", parse_dates=['time_hour'], index_col=0)
__planes = __read("planes")
__weather = __read("weather", parse_dates=['time_hour'])
__airlines = __read("airlines", index_col=0)

def get_airports() -> DataFrame:
    """Returns the airports datas

    Returns:
        A DataFrame containing the airports datas.
    """

    return __airports

def get_flights() -> DataFrame:
    """Returns the airports datas

    Returns:
        A DataFrame containing the airports datas.
    """

    return __flights

def get_planes() -> DataFrame:
    """Returns the planes datas

    Returns:
        A DataFrame containing the planes datas.
    """

    return __planes

def get_weather() -> DataFrame:
    """Returns the weather datas

    Returns:
        A DataFrame containing the weather datas.
    """

    return __weather

def get_airlines() -> DataFrame:
    """Returns the airlines datas

    Returns:
        A DataFrame containing the airlines datas.
    """

    return __airlines
