from QuBBD.utils import read
from QuBBD.constants import Constants
from os import path
from collections.abc import Iterable
import numpy as np


def get_full_data():
    filepath = path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES['qubbd'], Constants.DIRECTORIES["data"],
                         Constants.FILES["qubbdv3"])
    return read(filepath)


def get_stage_data(data, data_type="state", stage=1):
    state_columns_stage_data = Constants.get_data_dict()[data_type]
    if stage not in state_columns_stage_data:
        return np.empty([0,0])
    state_columns = state_columns_stage_data[stage]
    req_columns = []
    for key, value in state_columns.items():
        if not isinstance(value, str) and isinstance(value, Iterable):
            for k,v in value.items():
                req_columns.append(v)
        else:
            req_columns.append(value)
    return data[req_columns]


def get_all_data(data, data_type):
    state_columns_stage_data = Constants.get_data_dict()[data_type]
    required_columns = []
    for stage, stage_columns in state_columns_stage_data.items():
        for key, value in stage_columns.items():
            if not isinstance(value, str) and isinstance(value, Iterable):
                for k, v in value.items():
                    required_columns.append(v)
            else:
                required_columns.append(value)
    return data[required_columns]


if __name__ == '__main__':
    data = get_full_data()
    state_0 = get_stage_data(data, "states", 0)
    print(state_0.shape)
    decision_1 = get_stage_data(data, "decisions", 1)
    print(decision_1.shape)
    state_all_state = get_all_data(data, "states")
    print(state_all_state.shape)
    decision_all = get_all_data(data, "decisions")
    print(decision_all.shape)

