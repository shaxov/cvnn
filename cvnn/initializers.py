import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.keras.initializers import Initializer
from cvnn import utils


class ComplexZeros(Initializer):

    """ Zero initializer for complex layer. """
    def __init__(self, dtype=tf.dtypes.complex64):
        """

        Parameters
        ----------
            dtype: tf.dtypes.Dtype, optional (default=tf.dtypes.complex64)
                Data type of generated matrices.
        """
        utils.check_complex_dtype(dtype)
        self.dtype = tf.dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        real_part = array_ops.zeros(shape, tf.dtypes.float64)
        imag_part = array_ops.zeros(shape, tf.dtypes.float64)
        return tf.dtypes.cast(tf.complex(real_part, imag_part), dtype)

    def get_config(self):
        return {"dtype": self.dtype.name}


class ComplexRandomNormal(Initializer):

    """ Random normal initializer for complex layer. """
    def __init__(self,
                 real_mean=0.0,
                 real_stddev=1.0,
                 imag_mean=0.0,
                 imag_stddev=1.0,
                 seed=None,
                 dtype=tf.dtypes.complex64):
        """

        Parameters
        ----------
            real_mean: float, optional (default=0.0)
                Mean value in normal distribution which is used to generate real part.
            real_stddev: float, optional (default=1.0)
                Standard deviation value in normal distribution which is used to generate real part.
            imag_mean: float, optional (default=0.0)
                Mean value in normal distribution which is used to generate imaginary part.
            imag_stddev: float, optional (default=1.0)
                Standard deviation value in normal distribution which is used to generate imaginary part.
            dtype: tf.dtypes.Dtype, optional (default=tf.dtypes.complex64)
                Data type of generated matrices.
        """
        utils.check_complex_dtype(dtype)
        self.real_mean = real_mean
        self.imag_mean = imag_mean
        self.real_stddev = real_stddev
        self.imag_stddev = imag_stddev
        self.seed = seed
        self.dtype = tf.dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        real_part = random_ops.random_normal(
            shape, self.real_mean, self.real_stddev, tf.dtypes.float64, seed=self.seed)
        imag_part = random_ops.random_normal(
            shape, self.imag_mean, self.imag_stddev, tf.dtypes.float64, seed=self.seed)
        return tf.dtypes.cast(tf.complex(real_part, imag_part), dtype)

    def get_config(self):
        return {
            "real_mean": self.real_mean,
            "real_stddev": self.real_stddev,
            "imag_mean": self.imag_mean,
            "imag_stddev": self.imag_stddev,
            "seed": self.seed,
            "dtype": self.dtype.name
        }


class ComplexRandomUniform(Initializer):

    """ Random uniform initializer for complex layer. """
    def __init__(self,
                 real_min_val=0.,
                 real_max_val=1.,
                 imag_min_val=0.,
                 imag_max_val=1.,
                 seed=None,
                 dtype=tf.dtypes.complex64):
        """

        Parameters
        ----------
            real_min_val: float, optional (default=0.0)
                Min value in uniform distribution which is used to generate real part.
            real_max_val: float, optional (default=1.0)
                Max value value in uniform distribution which is used to generate real part.
            imag_min_val: float, optional (default=0.0)
                Min value in uniform distribution which is used to generate imaginary part.
            imag_max_val: float, optional (default=1.0)
                Max value in uniform distribution which is used to generate imaginary part.
            dtype: tf.dtypes.Dtype, optional (default=tf.dtypes.complex64)
                Data type of generated matrices.
        """
        utils.check_complex_dtype(dtype)
        self.real_min_val = real_min_val
        self.real_max_val = real_max_val
        self.imag_min_val = imag_min_val
        self.imag_max_val = imag_max_val
        self.seed = seed
        self.dtype = tf.dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        real_part = random_ops.random_uniform(
            shape, self.real_min_val, self.real_max_val, tf.dtypes.float64, seed=self.seed)
        imag_part = random_ops.random_uniform(
            shape, self.imag_min_val, self.imag_max_val, tf.dtypes.float64, seed=self.seed)
        initial_value = tf.dtypes.cast(tf.complex(real_part, imag_part), dtype)
        return initial_value

    def get_config(self):
        return {
            "real_min_val": self.real_min_val,
            "real_max_val": self.real_max_val,
            "imag_min_val": self.imag_min_val,
            "imag_max_val": self.imag_max_val,
            "seed": self.seed,
            "dtype": self.dtype.name
        }


class ComplexGlorotUniform(Initializer):

    """ Glorot uniform initializer for complex layer. """
    def __init__(self,
                 seed=None,
                 dtype=tf.dtypes.complex64):
        """

        Parameters
        ----------
            dtype: tf.dtypes.Dtype, optional (default=tf.dtypes.complex64)
                Data type of generated matrices.
        """
        utils.check_complex_dtype(dtype)
        self.real_init = tf.keras.initializers.GlorotUniform(seed=seed)
        self.imag_init = tf.keras.initializers.GlorotUniform(seed=seed)
        self.dtype = tf.dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        real_part = self.real_init(shape, tf.dtypes.float32)
        imag_part = self.imag_init(shape, tf.dtypes.float32)
        initial_value = tf.dtypes.cast(tf.complex(real_part, imag_part), dtype)
        return initial_value

    def get_config(self):
        return {
            "seed": self.seed,
            "dtype": self.dtype.name
        }


def get(identifier):
    if not isinstance(identifier, str):
        return identifier
    if identifier == "complex_zeros":
        return ComplexZeros
    elif identifier == "complex_random_normal":
        return ComplexRandomNormal
    elif identifier == "complex_random_uniform":
        return ComplexRandomUniform
    elif identifier == "complex_glorot_uniform":
        return ComplexGlorotUniform
    else:
        raise ValueError("Invalid initializer identifier.")
