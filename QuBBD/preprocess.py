from QuBBD.utils import read
from QuBBD.constants import Constants
from os import path
from collections.abc import Iterable
import numpy as np
import pandas as pd

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
    required_columns = list(set(required_columns))
    return data[required_columns]


def encode_data(data, id_column="Dummy ID", save=False, save_loc=None):
    new_df = data[[id_column]].copy()
    columns = data.columns.values
    for column in columns:
        num_values = len(pd.unique(data[column].values.flatten()))
        if num_values < data.shape[0]/10:
            one_hot = pd.get_dummies(data[column], prefix=column)
            new_df = new_df.join(one_hot)
        elif column == "Affected Lymph node cleaned":
            values = data[column].values
            value_set = set()
            for value in values:
                if isinstance(value, str):
                    vals = [i.strip() for i in value.split(",")]
                    value_set = value_set.union(set(vals))
                else:
                    value_set.add("None")
            value_set = list(value_set)
            value_dict = {val:i for i, val in enumerate(value_set)}
            value_set = [column+"_"+i for i in value_set]
            one_hot = np.zeros((data.shape[0], len(value_set)))
            for i, value in enumerate(values):
                if isinstance(value, str):
                    for v in value.split(","):
                        one_hot[i, value_dict[v.strip()]] = 1
                else:
                    one_hot[i, value_dict["None"]] = 1
            one_hot_frame = pd.DataFrame(data=one_hot,
                                         columns=value_set)
            new_df = new_df.join(one_hot_frame)
        else:
            new_df[column] = data[column].copy()
        if save:
            new_df.to_csv(save_loc)
    return new_df


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

    state_data_loc = path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["qubbd"], Constants.DIRECTORIES["data"], Constants.FILES["processed_data_state"])
    decision_data_loc = path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["qubbd"], Constants.DIRECTORIES["data"], Constants.FILES["processed_data_decision"])
    all_state = encode_data(state_all_state, save=True, save_loc=state_data_loc)

