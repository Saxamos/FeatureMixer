import numpy as np
import sklearn.preprocessing as preproc


def dummy_encode_pandas_features(data_frame):
    label_encoder = preproc.LabelEncoder()
    columns_to_encode = list(data_frame.select_dtypes(include=['category', 'object']))
    numeric_data_frame = data_frame.copy()
    for feature in columns_to_encode:
        try:
            numeric_data_frame[feature] = label_encoder.fit_transform(data_frame[feature])
        except:
            print('Error encoding ' + feature)
    return numeric_data_frame


def extract_feature_from_matrix(matrix, feature_rank):
    return matrix[:, feature_rank]


def operation(operator, matrix, tuple_feature):
    feature_1 = extract_feature_from_matrix(matrix, tuple_feature[0])
    feature_2 = extract_feature_from_matrix(matrix, tuple_feature[1])
    if operator == 'addition':
        return feature_1 + feature_2
    if operator == 'multiplication':
        return np.multiply(feature_1, feature_2)


def add_feature_to_matrix(matrix, feature):
    shape = matrix.shape
    new_matrix = np.zeros((shape[0], shape[1] + 1))
    new_matrix[:, :-1] = matrix
    new_matrix[:, -1] = feature
    return new_matrix


def create_list_of_feature_number_to_apply_operation(shape_col):
    return [(i, j) for i in range(shape_col) for j in range(i, shape_col) if i != j]


def scale_features_from_pandas_to_numpy_matrix(data_frame):
    scaler = preproc.MinMaxScaler()
    return scaler.fit_transform(data_frame)


def scale_one_numpy_feature(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())


def feature_creation(data_frame, operator):
    dummy_data_frame = dummy_encode_pandas_features(data_frame)
    scaled_matrix = scale_features_from_pandas_to_numpy_matrix(dummy_data_frame)
    list_of_feature = create_list_of_feature_number_to_apply_operation(scaled_matrix.shape[1])
    for tuple_feature in list_of_feature:
        new_feature = operation(operator, scaled_matrix, tuple_feature)
        new_feature_scaled = scale_one_numpy_feature(new_feature)
        scaled_matrix = add_feature_to_matrix(scaled_matrix, new_feature_scaled)
    return scaled_matrix
