import numpy as np


def transform_pandas_to_numpy(data_frame):
    np_array = data_frame.as_matrix()
    return np_array


def extract_feature_from_matrix(matrix, feature_rank):
    return matrix[:, feature_rank]


def operation(operator, matrix, tuple_feature):
    feature_1 = extract_feature_from_matrix(matrix, tuple_feature[0])
    feature_2 = extract_feature_from_matrix(matrix, tuple_feature[1])
    if operator == "addition":
        return feature_1 + feature_2
    if operator == "multiplication":
        return feature_1 * feature_2


def add_feature_to_matrix(matrix, feature):
    shape = matrix.shape
    new_matrix = np.zeros((shape[0], shape[1] + 1))
    new_matrix[:, :-1] = matrix
    new_matrix[:, -1] = feature
    return new_matrix


def create_list_of_feature_number_to_apply_operation(shape_col):
    return [(i, j) for i in range(shape_col) for j in range(i, shape_col) if i != j]


def feature_creation(data_frame, operator):
    matrix = transform_pandas_to_numpy(data_frame)
    list_of_feature = create_list_of_feature_number_to_apply_operation(matrix.shape[1])
    for tuple_feature in list_of_feature:
        new_feature = operation(operator, matrix, tuple_feature)
        matrix = add_feature_to_matrix(matrix, new_feature)
    return matrix
