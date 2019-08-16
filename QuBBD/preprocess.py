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


def encode_state_data(data, id_column="Dummy ID", save=False, save_loc=None):
    new_df = data[[id_column]].copy()
    columns = data.columns.values

    state_columns = {column: [] for column in columns}

    for column in columns:
        num_values = len(pd.unique(data[column].values.flatten()))
        if num_values < data.shape[0]/10:
            one_hot = pd.get_dummies(data[column], prefix=column)
            state_columns[column] = list(one_hot)
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
            state_columns[column] = list(one_hot_frame)
            new_df = new_df.join(one_hot_frame)
        else:
            new_df[column] = data[column].copy()
            state_columns[column] = [column]
    if save:
        new_df.to_csv(save_loc)
    return new_df, state_columns


def encode_decision_data(data, id_column="Dummy ID", save=False, save_loc=None):
    new_df = data[[id_column]].copy()
    columns = data.columns.values

    decision_columns = {column: [] for column in columns}

    for column in columns:
        num_values = len(pd.unique(data[column].values.flatten()))
        if num_values < data.shape[0] / 10:
            one_hot = pd.get_dummies(data[column], prefix=column)
            decision_columns[column] = list(one_hot)
            new_df = new_df.join(one_hot)
        else:
            new_df[column] = data[column].copy()
            decision_columns[column] = [column]
    if save:
        new_df.to_csv(save_loc)
    return new_df, decision_columns


def create_data_vector(state_data, all_state_columns, id_column="Dummy ID"):
    n, m = state_data.shape
    state_columns = Constants.STATES
    num_states = len(state_columns.keys())

    x_train = np.zeros((n * num_states, m-1), dtype=float)

    state_col_dict = {i: [] for i in range(num_states)}

    for state_num, state_cols in state_columns.items():
        for ke, col in state_cols.items():
            if col==id_column:
                continue
            if isinstance(col, str):
                state_col_dict[state_num] += all_state_columns[col]
            else:
                for k,v in col.items():
                    state_col_dict[state_num] += all_state_columns[v]

    for i in range(n):
        start = 0
        for j in range(num_states):
            row_data = state_data.fillna(0)[state_col_dict[j]].iloc[[i]].values.flatten()
            l = len(row_data)
            if ">20" in row_data:
                row_data[row_data == ">20"] = 20
            x_train[(i*num_states)+j, start:start+l] = np.array(row_data)
            start += l
    return x_train


def create_decision_vector(decision_data, all_decision_columns, id_column="Dummy ID"):
    n, m = decision_data.shape
    decision_columns = Constants.DECISIONS
    num_decisions = decision_columns.keys()

    y_train = np.zeros((n * len(num_decisions), m-1))

    decision_col_dict = {i: [] for i in num_decisions}

    for decision_num, decision_cols in decision_columns.items():
        for ke, col in decision_cols.items():
            if col==id_column:
                continue
            if isinstance(col, str):
                decision_col_dict[decision_num] += all_decision_columns[col]
            else:
                for k,v in col.items():
                    decision_col_dict[decision_num] += all_decision_columns[v]

    for i in range(n):
        start = 0
        for j, dec in enumerate(num_decisions):
            row_data = decision_data.fillna(0)[decision_col_dict[dec]].iloc[[i]].values.flatten()
            l = len(row_data)
            y_train[(i*len(num_decisions))+j, start:start+l] = np.array(row_data)
            start += l
    return y_train


def generate_data(save=False):
    print("Reading data")
    data = get_full_data()
    print("Data read, creating vectors")
    state_all_state = get_all_data(data, "states")
    decision_all = get_all_data(data, "decisions")
    state_data_loc = path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["qubbd"],
                               Constants.DIRECTORIES["data"], Constants.FILES["processed_data_state"])
    decision_data_loc = path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES["qubbd"],
                                  Constants.DIRECTORIES["data"], Constants.FILES["processed_data_decision"])
    all_state, all_state_columns = encode_state_data(state_all_state, save=save, save_loc=state_data_loc)
    print("Created state vectors")
    all_decisions, all_decision_columns = encode_decision_data(decision_all, save=save, save_loc=decision_data_loc)
    print("Created decision vectors")
    x_train = create_data_vector(all_state, all_state_columns)
    y_train = create_decision_vector(all_decisions, all_decision_columns)
    return x_train, y_train


def create_one_state_data(state_data, all_state_columns, state=0, id_column="Dummy ID"):
    if state not in Constants.STATES:
        print("State not present")
        return None

    state_columns = Constants.STATES[state]

    state_col_list = []

    for key, col in state_columns.items():
        if col == id_column:
            continue
        if isinstance(col, str):
            state_col_list += all_state_columns[col]
        else:
            for k,v in col.items():
                state_col_list += all_state_columns[v]

    x_train = state_data[state_col_list].values

    x_train[x_train == ">20"] = 20  # For that one value that isn't corrected in pycharm cache

    return x_train.astype(np.float)


def create_one_decision_data(decision_data, all_decision_columns, decision=1, id_column="Dummy ID"):
    if decision not in Constants.DECISIONS:
        print("Decision not present")
        return None

    decision_columns = Constants.DECISIONS[decision]

    decision_col_list = []

    for key, col in decision_columns.items():
        if col == id_column:
            continue
        if isinstance(col, str):
            decision_col_list += all_decision_columns[col]
        else:
            for k,v in col.items():
                decision_col_list += all_decision_columns[v]

    return decision_data[decision_col_list].values.astype(np.uint)


def generate_one_state_data(state=0):
    data = get_full_data()

    state_all_state = get_all_data(data, "states")
    decision_all = get_all_data(data, "decisions")

    all_state, all_state_columns = encode_state_data(state_all_state)
    all_decisions, all_decision_columns = encode_decision_data(decision_all)

    x_train = create_one_state_data(all_state, all_state_columns, state=state)
    y_train = create_one_decision_data(all_decisions, all_decision_columns, decision=state+1)

    return x_train, y_train


def test_funcs():
    data = get_full_data()
    state_0 = get_stage_data(data, "states", 0)
    print(state_0.shape)
    decision_1 = get_stage_data(data, "decisions", 1)
    print(decision_1.shape)
    state_all_state = get_all_data(data, "states")
    print(state_all_state.shape)
    decision_all = get_all_data(data, "decisions")
    print(decision_all.shape)


if __name__ == '__main__':
    x_train, y_train = generate_data()
    print(x_train.shape, y_train.shape)

    # x_train, y_train = generate_one_state_data(2)
    # print(x_train.shape, y_train.shape)
