import pandas as pd
import numpy as np
from os import path, makedirs


def read(filepath, encoding='ISO-8859-1', sep=",", dtype=str):
    '''
    Function to read csv file
    :param filepath: <class 'string'> location string
    :param encoding: <class 'string'> encoding to read csv
    :param sep: <class 'string'> separator
    :param dtype: <class 'string'> data type to read the csv file as
    :return: <class 'pandas.core.frame.DataFrame'> data frame containing the csv data
    '''
    return pd.read_csv(filepath, encoding=encoding, sep=sep, dtype=dtype)



