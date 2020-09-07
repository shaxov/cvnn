import unittest
import activation
import numpy as np


class TestActivation(unittest.TestCase):

    def test_complex_relu(self):
        np.random.seed(42)
        relu = activation.ComplexReLU()
        x = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(10, 10))).astype('complex64')
        y = relu(x)
        x_real, x_imag = np.real(x), np.imag(x)
        mask_x_real = x_real < 0
        mask_x_imag = x_imag < 0
        y_real, y_imag = np.real(y), np.imag(y)
        mask_y_real = y_real == 0
        mask_y_imag = y_imag == 0
        np.testing.assert_array_equal(mask_x_real, mask_y_real)
        np.testing.assert_array_equal(mask_x_imag, mask_y_imag)
