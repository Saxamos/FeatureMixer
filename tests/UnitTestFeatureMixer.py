import pandas as pd
import pytest

from app.FeatureMixer import *


def test_extract_feature_0_from_numpy_matrix():
    # Given
    np_matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    feature_rank = 0
    expected = [[0], [0], [0], [2]]

    # When
    result = extract_feature_from_matrix(np_matrix, feature_rank)

    # Then
    assert (result == expected).all()


@pytest.mark.skip # mocker pour devenir TU + faire celui pour autres opérations
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


@pytest.mark.skip # a reparer ou cas a traiter
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


def test_scale_the_features_from_pandas():
    # Given
    pandas_data_frame = pd.DataFrame(data={
        'feature_1': [0, 0, 3],
        'feature_2': [2, 0, 1],
        'feature_3': [4, 6, 2]
    })
    expected = np.matrix([
        [0, 1, 0.5],
        [0, 0, 1],
        [1, 0.5, 0]
    ])

    # When
    result = scale_features_from_pandas_to_numpy_matrix(pandas_data_frame)

    # Then
    assert (result == expected).all()


def test_scale_the_created_feature():
    # Given
    np_matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    feature_created = np.array([6, 2, 4])
    expected = np.array([1, 0, 0.5])

    # When
    result = scale_one_numpy_feature(feature_created)

    # Then
    assert (result == expected).all()

# TODO :
#         - gérer les noms des features
#         - selection (1. par correlation 2. par importance: FI?)
#         - biner les features
#         - donner le FI classement
#         - reprendre les tests skip
#         - ajout d'opérations
#         - ajouter de la visu ? PCA, feature non supervisées
