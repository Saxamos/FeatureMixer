import numpy as np
import pandas as pd
import pytest

from app.FeatureMixer import transform_pandas_to_numpy, operation, extract_feature_from_matrix, feature_creation, \
    add_feature_to_matrix, create_list_of_feature_number_to_apply_operation, dummy_encode_pandas_features


def test_transform_pandas_frame_into_numpy_matrix():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': [0, 0, 0, 2],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5]
    })
    expected_np_matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])

    # When
    result_matrix = transform_pandas_to_numpy(pandas_data_frame)

    # Then
    assert (result_matrix == expected_np_matrix).all()


def test_extract_feature_0_from_numpy_matrix():
    # Given
    np_matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    feature_rank = 0
    expected = np.array([0, 0, 0, 2])

    # When
    result = extract_feature_from_matrix(np_matrix, feature_rank)

    # Then
    for i in range(4):
        assert result[i] == expected[i]


def test_operation_addition_add_features():
    # Given
    operator = 'addition'
    matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    tuple_feature = (1, 2)
    expected = np.array([5, 2, 4, 8])

    # When
    result = operation(operator, matrix, tuple_feature)

    # Then
    for i in range(4):
        assert result[i] == expected[i]


def test_append_new_feature_to_matrix():
    # Given
    matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    feature = np.array([1, 1, 2, 5])
    expected = np.matrix([
        [0, 1, 4, 1],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
        [2, 3, 5, 5]
    ])

    # When
    result = add_feature_to_matrix(matrix, feature)

    # Then
    assert (result == expected).all()


def test_create_tuple_of_col_to_apply_operation():
    # Given
    shape_col = 3
    expected = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]

    # When
    result = create_list_of_feature_number_to_apply_operation(shape_col)

    # Then
    assert result == expected


def test_create_all_new_feature_combination_with_addition():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': [0, 0, 0, 2],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5]
    })
    expected = np.matrix([
        [0, 1, 4, 1, 4, 5],
        [0, 1, 1, 1, 1, 2],
        [0, 2, 2, 2, 2, 4],
        [2, 3, 5, 5, 7, 8]
    ])

    # When
    result = feature_creation(pandas_data_frame, 'addition')

    # Then
    assert (result == expected).all()


def test_create_all_new_feature_combination_with_multiplication():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': [0, 0, 0, 2],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5]
    })
    expected = np.matrix([
        [0, 1, 4, 0, 0, 4],
        [0, 1, 1, 0, 0, 1],
        [0, 2, 2, 0, 0, 4],
        [2, 3, 5, 6, 10, 15]
    ])

    # When
    result = feature_creation(pandas_data_frame, 'multiplication')

    # Then
    assert (result == expected).all()


def test_one_encode_the_feature_of_strings_in_pandas_data_frame():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': ['A', 'A', 'B', 'D'],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5],
    })
    expected_data_frame = pd.DataFrame(data={
        'feature_1': [0, 0, 1, 2],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5],
    })

    # When
    result_data_frame = dummy_encode_pandas_features(pandas_data_frame)

    # Then
    for col in pandas_data_frame.columns:
        assert (result_data_frame[col] == expected_data_frame[col]).all()


@pytest.mark.skip
def test_get_error_when_one_encode_feature_semi_string_semi_int():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': ['A', 'A', 'B', 'D'],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5],
        'feature_4': ['R', 1, 2, 'S'],
    })
    expected_data_frame = pd.DataFrame(data={
        'feature_1': [0, 0, 1, 2],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5],
        'feature_4': [0, 1, 2, 3],
    })

    # When
    result_data_frame = dummy_encode_pandas_features(pandas_data_frame)

    # Then
    assert print('blabla')


def test_create_all_feature_combination_with_multiplication_from_non_binary_data_frame():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': [2, 1, 2, 0],
        'feature_2': ['A', 'A', 'B', 'C'],
        'feature_3': [4, 1, 2, 5]
    })
    expected = np.matrix([
        [2, 0, 4, 0, 8, 0],
        [1, 0, 1, 0, 1, 0],
        [2, 1, 2, 2, 4, 2],
        [0, 2, 5, 0, 0, 10]
    ])

    # When
    result = feature_creation(pandas_data_frame, 'multiplication')

    # Then
    assert (result == expected).all()


# TODO :
#         - normaliser les features
#         - selection (1. par correlation 2. par importance: FI?)
#         - biner les features
#         - donner le FI classement
#         - reprendre les tests skip
#         - ajout d'opérations
