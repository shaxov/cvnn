import unittest
from cvnn import initializer
import numpy as np
import tensorflow as tf


class TestInitializer(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)

    def test_complex_zero_initializer(self):
        complex_zero_initializer = initializer.ComplexZeros()
        init = complex_zero_initializer(shape=(10, 10))
        self.assertEqual(init.numpy().shape, (10, 10))
        self.assertTrue(np.issubdtype(init.numpy().dtype, np.complexfloating))
        target = np.zeros(shape=(10, 10))
        np.testing.assert_almost_equal(target, np.real(init.numpy()))
        np.testing.assert_almost_equal(target, np.imag(init.numpy()))

    def test_complex_random_uniform_initializer(self):
        complex_random_uniform_initializer = initializer.ComplexRandomUniform()
        init = complex_random_uniform_initializer(shape=(10, 10))
        self.assertEqual(init.numpy().shape, (10, 10))
        self.assertTrue(np.issubdtype(init.numpy().dtype, np.complexfloating))
        self.assertTrue(np.real(init.numpy()).max() <= 1.)
        self.assertTrue(np.real(init.numpy()).min() >= 0.)
        self.assertTrue(np.imag(init.numpy()).max() <= 1.)
        self.assertTrue(np.imag(init.numpy()).min() >= 0.)

    def test_complex_random_normal_initializer(self):
        complex_random_uniform_initializer = initializer.ComplexRandomNormal()
        init = complex_random_uniform_initializer(shape=(100, 100))
        self.assertEqual(init.numpy().shape, (100, 100))
        self.assertTrue(np.issubdtype(init.numpy().dtype, np.complexfloating))
        self.assertAlmostEqual(np.real(init.numpy()).mean(), -0.0028726584)
        self.assertAlmostEqual(np.real(init.numpy()).std(), 0.99106705)
        self.assertAlmostEqual(np.imag(init.numpy()).mean(), -0.0050794417)
        self.assertAlmostEqual(np.imag(init.numpy()).std(), 0.99318254)


