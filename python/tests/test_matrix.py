"""Tests for matrix conversion utilities."""

import ctypes

import numpy as np
import pytest

from srukf._bindings import lib
from srukf.utils import (
    numpy_to_srukf_mat,
    srukf_mat_to_numpy,
    validate_square,
    validate_vector,
)


class TestNumpyToSrukfMat:
    def test_identity(self):
        arr = np.eye(3)
        mat_p = numpy_to_srukf_mat(arr)
        try:
            result = srukf_mat_to_numpy(mat_p)
            np.testing.assert_array_equal(result, arr)
        finally:
            lib.srukf_mat_free(mat_p)

    def test_vector(self):
        arr = np.array([1.0, 2.0, 3.0])
        mat_p = numpy_to_srukf_mat(arr)
        try:
            result = srukf_mat_to_numpy(mat_p)
            assert result.shape == (3, 1)
            np.testing.assert_array_equal(result.ravel(), arr)
        finally:
            lib.srukf_mat_free(mat_p)

    def test_rectangular(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mat_p = numpy_to_srukf_mat(arr)
        try:
            result = srukf_mat_to_numpy(mat_p)
            np.testing.assert_array_equal(result, arr)
        finally:
            lib.srukf_mat_free(mat_p)

    def test_column_major_layout(self):
        """Verify the C matrix stores data in column-major order."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mat_p = numpy_to_srukf_mat(arr)
        try:
            mat = mat_p.contents
            # Column-major: data should be [1, 3, 5, 2, 4, 6]
            assert mat.data[0] == 1.0
            assert mat.data[1] == 3.0
            assert mat.data[2] == 5.0
            assert mat.data[3] == 2.0
            assert mat.data[4] == 4.0
            assert mat.data[5] == 6.0
        finally:
            lib.srukf_mat_free(mat_p)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            numpy_to_srukf_mat(np.zeros((2, 3, 4)))

    def test_int_input_converted(self):
        arr = np.array([[1, 2], [3, 4]])
        mat_p = numpy_to_srukf_mat(arr)
        try:
            result = srukf_mat_to_numpy(mat_p)
            assert result.dtype == np.float64
            np.testing.assert_array_equal(result, arr.astype(np.float64))
        finally:
            lib.srukf_mat_free(mat_p)


class TestSrukfMatToNumpy:
    def test_copy_is_independent(self):
        arr = np.eye(2)
        mat_p = numpy_to_srukf_mat(arr)
        try:
            result = srukf_mat_to_numpy(mat_p, copy=True)
            # Modifying result should not affect C memory
            result[0, 0] = 999.0
            result2 = srukf_mat_to_numpy(mat_p, copy=True)
            assert result2[0, 0] == 1.0
        finally:
            lib.srukf_mat_free(mat_p)

    def test_view_shares_memory(self):
        arr = np.eye(2)
        mat_p = numpy_to_srukf_mat(arr)
        try:
            view = srukf_mat_to_numpy(mat_p, copy=False)
            # view.base should not be None (it's a view)
            assert view.base is not None
        finally:
            lib.srukf_mat_free(mat_p)

    def test_null_raises(self):
        from srukf._bindings import SrukfMat, POINTER
        null = ctypes.POINTER(SrukfMat)()
        with pytest.raises(ValueError, match="NULL"):
            srukf_mat_to_numpy(null)


class TestRoundTrip:
    """Verify data survives numpy -> C -> numpy conversion."""

    @pytest.mark.parametrize(
        "shape",
        [(1, 1), (3, 3), (5, 2), (2, 7), (10, 10)],
    )
    def test_random_matrix(self, shape):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(shape)
        mat_p = numpy_to_srukf_mat(arr)
        try:
            result = srukf_mat_to_numpy(mat_p)
            np.testing.assert_array_almost_equal(result, arr)
        finally:
            lib.srukf_mat_free(mat_p)


class TestValidation:
    def test_validate_square_ok(self):
        result = validate_square(np.eye(3), "test")
        assert result.shape == (3, 3)

    def test_validate_square_non_square(self):
        with pytest.raises(ValueError, match="square"):
            validate_square(np.ones((2, 3)), "test")

    def test_validate_square_1d(self):
        with pytest.raises(ValueError, match="square"):
            validate_square(np.array([1.0, 2.0]), "test")

    def test_validate_vector_ok(self):
        result = validate_vector(np.array([1.0, 2.0, 3.0]), 3, "test")
        assert result.shape == (3,)

    def test_validate_vector_wrong_length(self):
        with pytest.raises(ValueError, match="3 elements"):
            validate_vector(np.array([1.0, 2.0]), 3, "test")

    def test_validate_vector_2d_flattened(self):
        result = validate_vector(np.array([[1.0], [2.0]]), 2, "test")
        assert result.shape == (2,)
