import numpy as np
import pandas as pd

from app.FeatureMixer import transform_pandas_to_numpy, operation, extract_feature_from_matrix, feature_creation, \
    add_feature_to_matrix, create_list_of_feature_number_to_apply_operation


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
    operator = "addition"
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