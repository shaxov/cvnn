import unittest
import layer
import numpy as np


class TestLayer(unittest.TestCase):

    def setUp(self):
        self.input_dim = 20
        self.output_dim = 10
        self.dropout = 0.5

    def test_complex_dense_kernel_bias(self):
        dense = layer.ComplexDense(self.output_dim)
        x = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(1, self.input_dim)))
        _ = dense(x)
        self.assertEqual(dense.kernel.numpy().shape, (self.input_dim, self.output_dim))
        self.assertEqual(dense.bias.numpy().shape, (self.output_dim,))
        self.assertTrue(np.issubdtype(dense.kernel.dtype.as_numpy_dtype, np.complexfloating))
        self.assertTrue(np.issubdtype(dense.bias.dtype.as_numpy_dtype, np.complexfloating))

    def test_complex_dense_output(self):
        dense = layer.ComplexDense(self.output_dim)
        x = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(1, self.input_dim)))
        y = dense(x)
        self.assertEqual(y.numpy().shape, (1, self.output_dim))
        self.assertTrue(np.issubdtype(y.dtype.as_numpy_dtype, np.complexfloating))

    def test_complex_dropout_output(self):
        dropout = layer.ComplexDropout(self.dropout)
        x = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(1, self.input_dim)))
        y = dropout(x, training=True)
        self.assertTrue(np.sum(np.real(y.numpy()) == 0.) > 0)
        self.assertTrue(np.sum(np.imag(y.numpy()) == 0.) > 0)
        self.assertEqual(y.numpy().shape, (1, self.input_dim))
        self.assertTrue(np.issubdtype(y.dtype.as_numpy_dtype, np.complexfloating))


