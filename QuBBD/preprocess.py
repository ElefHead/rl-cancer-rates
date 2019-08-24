from QuBBD.utils import read
from QuBBD.constants import Constants
from os import path
from collections.abc import Iterable
import numpy as np
import pandas as pd


def get_full_data():
    '''
    Function to read and fetch QuBBD data as a dataframe
    QuBBD data resides as data_QuBBD_v3final.csv (to change name, see constants.py)
    :return: <class 'pandas.core.frame.DataFrame'> pandas dataframe containing QuBBD data
    '''
    filepath = path.join(Constants.DIRECTORIES["root"], Constants.DIRECTORIES['qubbd'], Constants.DIRECTORIES["data"],
                         Constants.FILES["qubbdv3"])
    return read(filepath)


def get_stage_data(data, data_type="state", stage=1):
    '''
    Specifically gets single "stage" of decision or state data.
    The stage refers to the timestep. S_0 -> D_1 -> S_1 -> ... -> S_t -> D_t+1
    :param data: <class 'pandas.core.frame.DataFrame'> dataframe containing Qubbd data
    :param data_type: <class 'string'> either "state" or "decisions"
    :param stage: <class 'integer'> timestep
    :return: <class 'pandas.core.frame.DataFrame'> pandas dataframe containing data from specific timestep
    '''
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
    '''
    Gets data from all timesteps for either states or decisions
    :param data: <class 'pandas.core.frame.DataFrame'> dataframe containing Qubbd data
    :param data_type: <class 'string'> either "state" or "decisions"
    :return: <class 'pandas.core.frame.DataFrame'> pandas dataframe containing state or decision data
    '''
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
    '''
    Function to one hot encode categorical state data
    :param data: <class 'pandas.core.frame.DataFrame'> dataframe containing Qubbd data
    :param id_column: <class 'string'> name of the id column in the dataset
            Used to avoid that column and create a new dataframe
    :param save: <class 'boolean'> Set true to save data as a .csv file
    :param save_loc: <class 'string'> Location to save the data
    :return: (<class 'pandas.core.frame.DataFrame'>, <class 'dict'>) A tuple containing a dataframe
            of one hot encoded state data and a dictionary containing states mapped with their one hot encoded
            state names
            Eg. if column gets encoded as column_yes and column_no,
            the dictionary will contain column: [column_yes, column_no]
    '''
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
    '''
    Function to one hot encode categorical decision data
    :param data: <class 'pandas.core.frame.DataFrame'> dataframe containing Qubbd data
    :param id_column: <class 'string'> name of the id column in the dataset
            Used to avoid that column and create a new dataframe
    :param save: <class 'boolean'> Set true to save data as a .csv file
    :param save_loc: <class 'string'> Location to save the data
    :return: (<class 'pandas.core.frame.DataFrame'>, <class 'dict'>) A tuple containing a dataframe
            of one hot encoded decision data and a dictionary containing decision mapped with their one hot encoded
            decision column names
            Eg. if column gets encoded as column_yes and column_no,
            the dictionary will contain column: [column_yes, column_no]
    '''
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
    '''
    Function to vectorize state data
    :param state_data: <class 'pandas.core.frame.DataFrame'> dataframe containing one hot encoded state data
    :param all_state_columns: <class 'dict'> a dictionary containing decision mapped with their one hot encoded
            state column names
    :param id_column: <class 'string'> name of the id column in the dataset
            Used to avoid that column and create a new dataframe
    :return: <class 'numpy.ndarray'> vectorized state data
    '''
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


def create_cross_product_decision_data(decision_data, num_decisions, decision_col_dict, id_column="Dummy ID"):
    '''
    Function to produce cross product of one hot encoded decision data
    :param decision_data: <class 'pandas.core.frame.DataFrame'> dataframe containing one hot encoded decision data
    :param num_decisions: <class 'list'> list of decision timesteps
    :param decision_col_dict: <class 'dict'> dict of decision column names mapped with their one hot encoded names
    :param id_column: <class 'str'> id column name, to avoid or to create new df
    :return: (<class 'dict'>, <class 'pandas.core.frame.DataFrame'>) dict containing new column names mapped with the timestep,
            new dataframe containing cross product data
    '''
    new_df = decision_data[[id_column]].copy()

    n, m = decision_data.shape
    new_decision_col_dict = {i: [] for i in num_decisions}

    # for decision_num, decision_cols in decision_columns.items():
    for num, cols in decision_col_dict.items():
        len_cols = len(cols)
        for i in range(len_cols):
            for j in range(i+1, len_cols):
                col_name = "{}cross{}".format(cols[i], cols[j])
                new_decision_col_dict[num].append(col_name)
                new_df[col_name] = np.nan

    for i in range(n):
        for state, col in new_decision_col_dict.items():
            for c in col:
                col1, col2 = c.split("cross")
                if int(decision_data[col1][i]) == 1 and int(decision_data[col2][i]) == 1:
                    new_df.at[i, c] = 1

    return new_decision_col_dict, new_df


def create_decision_vector(decision_data, all_decision_columns, id_column="Dummy ID", cross_product=True):
    '''
    Function to vectorize decision data
    :param decision_data: <class 'pandas.core.frame.DataFrame'> dataframe containing one hot encoded decision data
    :param all_decision_columns: <class 'dict'> a dictionary containing decision mapped with their one hot encoded
            decision column names
    :param id_column: <class 'string'> name of the id column in the dataset
            Used to avoid that column and create a new dataframe
    :return: <class 'numpy.ndarray'> vectorized decision data
    '''
    n, m = decision_data.shape
    decision_columns = Constants.DECISIONS
    num_decisions = decision_columns.keys()

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

    if cross_product:
        decision_col_dict, new_df = create_cross_product_decision_data(decision_data, num_decisions, decision_col_dict)
        m = new_df.shape[1]

    y_train = np.zeros((n * len(num_decisions), m - 1))

    for i in range(n):
        start = 0
        for j, dec in enumerate(num_decisions):
            if cross_product:
                row_data = new_df.fillna(0)[decision_col_dict[dec]].iloc[[i]].values.flatten()
            else:
                row_data = decision_data.fillna(0)[decision_col_dict[dec]].iloc[[i]].values.flatten()
            l = len(row_data)
            y_train[(i*len(num_decisions))+j, start:start+l] = np.array(row_data)
            start += l
    return y_train


def generate_data(save=False):
    '''
    One function to get data after all preprocessing
    :param save: <class 'boolean'> save or not
    :return: (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)vectorized data as training and testing
    '''
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
    '''
    Function to vectorize one timestep state data
    :param state_data: <class 'pandas.core.frame.DataFrame'> dataframe containing one hot encoded state data
    :param all_state_columns: <class 'dict'> a dictionary containing decision mapped with their one hot encoded
            state column names
    :param state: <class 'integer'> timestep
    :param id_column: <class 'string'> name of the id column in the dataset
            Used to avoid that column and create a new dataframe
    :return: <class 'numpy.ndarray'> vectorized state data
    '''
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
    '''
    Function to vectorize one timestep state data
    :param decision_data: <class 'pandas.core.frame.DataFrame'> dataframe containing one hot encoded decision data
    :param all_decision_columns: <class 'dict'> a dictionary containing decision mapped with their one hot encoded
            decision column names
    :param decision: <class 'integer'> timestep
    :param id_column: <class 'string'> name of the id column in the dataset
            Used to avoid that column and create a new dataframe
    :return: <class 'numpy.ndarray'> vectorized decision data
    '''
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
    '''
    Function to generate one timestep vectorized training and testing data
    :param state: <class 'integer'> timestep (refers to state timestep)
    :return: (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>) one timestep training and testing data
    '''
    data = get_full_data()

    state_all_state = get_all_data(data, "states")
    decision_all = get_all_data(data, "decisions")

    all_state, all_state_columns = encode_state_data(state_all_state)
    all_decisions, all_decision_columns = encode_decision_data(decision_all)

    x_train = create_one_state_data(all_state, all_state_columns, state=state)
    y_train = create_one_decision_data(all_decisions, all_decision_columns, decision=state+1)

    return x_train, y_train


def isolate_x_by_single_decision(x_train, y_train, decision_index):
    '''
    Function to get training data that lead to one decision
    :param x_train: <class 'numpy.ndarray'> training data vectors
    :param y_train: <class 'numpy.ndarray'> output (decision) data vectors
    :param decision_index: <class 'integer'> output index to isolate
    :return: (<class 'numpy.ndarray'>, <class 'list'>) isolated training vectors and list containing indices
    '''
    indices = np.argwhere(y_train[:, decision_index] == 1).flatten()
    return x_train[indices, :], indices


def count_non_zero_decisions(x_train, y_train, threshold=None):
    '''
    Function to check if some decision has ever been taken
    :param x_train: <class 'numpy.ndarray'> training data vectors
    :param y_train: <class 'numpy.ndarray'> output (decision) data vectors
    :param threshold: <class 'integer'> set threshold value to see if a decision has been taken x number of times
    :return:
    '''
    count = 0
    for i in range(y_train.shape[1]):
        reduced_x, indices = isolate_x_by_single_decision(x_train, y_train, i)
        num_indices = len(indices)
        if num_indices:
            if threshold is not None:
                count += 1 if num_indices > threshold else 0
            else:
                count += 1
    return count


def get_next_states(x_train, indices):
    '''
    Function to get the succeeding states. Used for linear model creation.
    :param x_train: <class 'numpy.ndarray'> training data vectors
    :param indices: <class 'list'> list of indices
    :return: <class 'numpy.ndarray'> matrix containing succeeding state vectors
    '''
    return x_train[indices + 1]


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
    print(type(get_full_data()))

    x_train, y_train = generate_data()
    print(x_train.shape, y_train.shape)

    reduced_x, indices = isolate_x_by_single_decision(x_train, y_train, 0)
    print(reduced_x.shape)
