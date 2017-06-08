import pandas as pd

from app.FeatureMixer import *


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
    expected = [[5], [2], [4], [8]]

    # When
    result = operation(operator, matrix, tuple_feature)

    # Then
    assert (result == expected).all()


def test_operation_multiplication_multiplicate_features():
    # Given
    operator = 'multiplication'
    matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    tuple_feature = (1, 2)
    expected = [[4], [1], [4], [15]]

    # When
    result = operation(operator, matrix, tuple_feature)

    # Then
    assert (result == expected).all()


def test_create_all_new_feature_combination_with_addition():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': [0, 0, 0, 2],
        'feature_2': [1, 1, 2, 3],
        'feature_3': [4, 1, 2, 5]
    })
    expected = np.matrix([
        [0, 0.0, 0.75, 0.00, 0.375, 0.375],
        [0, 0.0, 0.00, 0.00, 0.000, 0.000],
        [0, 0.5, 0.25, 0.25, 0.125, 0.375],
        [1, 1.0, 1.00, 1.00, 1.000, 1.000]
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
        [0, 0.0, 0.75, 0, 0, 0.000],
        [0, 0.0, 0.00, 0, 0, 0.000],
        [0, 0.5, 0.25, 0, 0, 0.125],
        [1, 1.0, 1.00, 1, 1, 1.000]
    ])

    # When
    result = feature_creation(pandas_data_frame, 'multiplication')

    # Then
    assert (result == expected).all()


def test_create_all_feature_combination_with_multiplication_from_non_numeric_data_frame():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': [2, 1, 2, 0],
        'feature_2': ['A', 'A', 'B', 'C'],
    })
    expected = np.matrix([
        [1.0, 0.0, 0],
        [0.5, 0.0, 0],
        [1.0, 0.5, 1],
        [0.0, 1.0, 0]
    ])

    # When
    result = feature_creation(pandas_data_frame, 'multiplication')

    # Then
    assert (result == expected).all()