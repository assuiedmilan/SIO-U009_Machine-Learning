"""Main module for learning module 18"""

import math

import pandas as pd
from pandas import DataFrame

from sio_u009.exercices.module18.data_tools import extract_test_and_training_set
from sio_u009.exercices.module18.data_tools import get_all_flights_from_in_the_last_months
from sio_u009.exercices.module18.data_tools import join_new_data_using_common_data
from sio_u009.exercices.module18.datas import get_planes
from sio_u009.exercices.module18.datas import get_weather
from sio_u009.exercices.module18.simple_linear_regression import SimpleLinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def main():
    """Main function"""

    flights_datas = prepare_data()
    _, training_set = extract_test_and_training_set(flights_datas, 20)

    simple_regression = SimpleLinearRegression(training_set)
    simple_regression.adjust_and_plot_model_for('arr_delay', 'dep_delay')

    simple_regression.model.summary()
    simple_regression.print_variability_percentage()
    simple_regression.get_mean_quadratic_error()


def prepare_data() -> DataFrame:
    """The objective of this function is to:

    Returns all flights that departed from Boston in the last seven months.

    For each flight, include:
     - the number of seats on board
     - the weather data: dew point (temperature at which humidity condense into solid water), relative humidity, winds direction/speed, precipitation, pressure and visibility)
     - transformed time data: which day of the week the flight took place (Monday - Sunday), was it a week end,was it during evening rush hour. This data is not present in the table but computed from it
     - transformed weather information: head/side wind speed (relative to the north, not aircraft direction - named cos/sin wind speed ), precipitation indication (are precipitation over 0)

    Notes:
        All flights with missing data will be dropped

    Returns:
        A DataFrame
    """
    boston_flights = get_all_flights_from_in_the_last_months('BOS', 7)
    boston_flights = join_new_data_using_common_data(boston_flights, get_planes(), ['tailnum'], ['seats'])
    boston_flights = join_new_data_using_common_data(boston_flights, get_weather(), ['origin', 'time_hour'], ['dewp', 'humid', 'wind_dir', 'wind_speed', 'precip', 'pressure', 'visib'])

    boston_flights['week_day'] = boston_flights.apply(lambda row: row.time_hour.strftime('%A'), axis=1)
    boston_flights['wknd'] = (boston_flights['week_day'] == 'Saturday') | (boston_flights['week_day'] == 'Sunday')
    boston_flights['evening_rush_hour'] = boston_flights['hour'].isin([17, 18])
    boston_flights['wind_sin'] = boston_flights.apply(lambda row: row.wind_speed * math.sin(row.wind_dir * math.pi / 180), axis=1)
    boston_flights['wind_cos'] = boston_flights.apply(lambda row: row.wind_speed * math.cos(row.wind_dir * math.pi / 180), axis=1)
    boston_flights['precip_indic'] = (boston_flights['precip'] > 0)

    boston_flights = boston_flights.dropna()

    return boston_flights

if __name__ == "__main__":
    main()
