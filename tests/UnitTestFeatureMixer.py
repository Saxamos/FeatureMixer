import numpy as np
import pandas as pd
import pytest

from app.FeatureMixer import (_operation, _add_feature_to_matrix_if_not_too_correlated,
                              _get_number_of_column, _are_feature_too_correlated,
                              _scale_one_numpy_feature, _scale_features_from_pandas_to_numpy_matrix,
                              _dummy_encode_pandas_features,
                              _create_list_of_feature_number_to_apply_operation,
                              _extract_feature_from_matrix)


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
    result = _extract_feature_from_matrix(np_matrix, feature_rank)

    # Then
    assert (result == expected).all()


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
    result = _operation(operator, matrix, tuple_feature)

    # Then
    assert (result == expected).all()


def test_append_new_feature_not_too_correlated_to_matrix():
    # Given
    matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    feature = np.array([1, 1, 9, 9])
    expected = np.matrix([
        [0, 1, 4, 1],
        [0, 1, 1, 1],
        [0, 2, 2, 9],
        [2, 3, 5, 9]
    ])
    list_feature_name = ['feature_1']

    # When
    result, list_feature_name = _add_feature_to_matrix_if_not_too_correlated(matrix, feature,
                                                                             list_feature_name)

    # Then
    assert (result == expected).all()


def test_add_feature_to_matrix_if_not_too_correlated_append_new_variable_name_to_column_list():
    # Given
    list_feature_name = ['feature_1']
    expected_new_feature_name = ['feature_1', 'feature_1*feature_2']
    matrix = np.matrix([[0], [0], [0], [2]])
    feature = np.array([1, 1, 9, 9])

    # When
    matrix_result, new_feature_name = _add_feature_to_matrix_if_not_too_correlated(matrix,
                                                                                   feature,
                                                                                   list_feature_name)

    # Then
    assert new_feature_name == expected_new_feature_name


def test_don_t_append_new_feature_correlated_to_matrix():
    # Given
    matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    feature = np.array([1, 1, 1, 5])
    expected = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    list_feature_name = ['feature_1']

    # When
    result, list_feature_name = _add_feature_to_matrix_if_not_too_correlated(matrix, feature,
                                                                             list_feature_name)

    # Then
    assert (result == expected).all()


def test_get_number_of_column_in_matrix():
    # Given
    matrix = np.matrix([
        [0, 1, 4],
        [0, 1, 1],
        [0, 2, 2],
        [2, 3, 5]
    ])
    expected = 3

    # When
    result = _get_number_of_column(matrix)

    # Then
    assert result == expected


def test_create_tuple_of_col_to_apply_operation():
    # Given
    shape_col = 3
    expected = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]

    # When
    result = _create_list_of_feature_number_to_apply_operation(shape_col)

    # Then
    assert result == expected


def test_operation_addition_on_two_features():
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
    result = _operation(operator, matrix, tuple_feature)

    # Then
    assert (result == expected).all()


def test_operation_multiplication_one_two_features():
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
    result = _operation(operator, matrix, tuple_feature)

    # Then
    assert (result == expected).all()


@pytest.mark.skip  # a reparer ou cas a traiter
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
    result_data_frame = _dummy_encode_pandas_features(pandas_data_frame)

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
    result = _scale_features_from_pandas_to_numpy_matrix(pandas_data_frame)

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
    result = _scale_one_numpy_feature(feature_created)

    # Then
    assert (result == expected).all()


def test_return_true_if_features_are_too_correlated():
    # Given
    feature_1 = np.array([0, 2, 1])
    feature_2 = np.array([2, 4, 2.5])
    expected = True

    # When
    result = _are_feature_too_correlated(feature_1, feature_2)

    # Then
    assert result == expected


def test_return_true_if_features_are_too_negatively_correlated():
    # Given
    feature_1 = np.array([0, 2, 1])
    feature_2 = np.array([0, -2, -1])
    expected = True

    # When
    result = _are_feature_too_correlated(feature_1, feature_2)

    # Then
    assert result == expected


def test_return_false_if_features_are_not_too_correlated():
    # Given
    feature_1 = np.array([0, 2, 4])
    feature_2 = np.array([2, 4, 1])
    expected = False

    # When
    result = _are_feature_too_correlated(feature_1, feature_2)

    # Then
    assert result == expected

# TODO :
#         - $$kwargs pour mettre toutes les operations qu'on veut!!
#         - gérer les noms des features
#         - ajout de l'entropie mutuelle au lieu de corr
#   + retourner pandas ac nom a la fin
#         - selection (1. par correlation 2. par importance: FI?)
#         - feature aggregateur? et PCA ICA
#         - biner les features
#         - donner le FI classement
#         - reprendre les tests skip
#         - ajout d'opérations
#         - ajouter de la visu ? PCA, feature non supervisées
