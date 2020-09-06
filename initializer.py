from tensorflow import dtypes
from tensorflow import complex
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.keras.initializers import Initializer
import utils


class ComplexZeros(Initializer):

    def __init__(self, dtype=dtypes.complex64):
        utils.check_complex_dtype(dtype)
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        real_part = array_ops.zeros(shape, dtypes.float64)
        imag_part = array_ops.zeros(shape, dtypes.float64)
        return dtypes.cast(complex(real_part, imag_part), dtype)

    def get_config(self):
        return {"dtype": self.dtype.name}


class ComplexRandomNormal(Initializer):
    def __init__(self,
                 real_mean=0.0,
                 real_stddev=1.0,
                 imag_mean=0.0,
                 imag_stddev=1.0,
                 seed=None,
                 dtype=dtypes.complex64):
        utils.check_complex_dtype(dtype)
        self.real_mean = real_mean
        self.imag_mean = imag_mean
        self.real_stddev = real_stddev
        self.imag_stddev = imag_stddev
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        real_part = random_ops.random_normal(
            shape, self.real_mean, self.real_stddev, dtypes.float64, seed=self.seed)
        imag_part = random_ops.random_normal(
            shape, self.imag_mean, self.imag_stddev, dtypes.float64, seed=self.seed)
        return dtypes.cast(complex(real_part, imag_part), dtype)

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
    def __init__(self,
                 real_min_val=0,
                 real_max_val=None,
                 imag_min_val=0,
                 imag_max_val=None,
                 seed=None,
                 dtype=dtypes.complex64):
        utils.check_complex_dtype(dtype)
        self.real_min_val = real_min_val
        self.real_max_val = real_max_val
        self.imag_min_val = imag_min_val
        self.imag_max_val = imag_max_val
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        real_part = random_ops.random_uniform(
            shape, self.real_min_val, self.real_max_val, dtypes.float64, seed=self.seed)
        imag_part = random_ops.random_uniform(
            shape, self.imag_min_val, self.imag_max_val, dtypes.float64, seed=self.seed)
        initial_value = dtypes.cast(complex(real_part, imag_part), dtype)
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


def get(identifier):
    if identifier == "complex_zeros":
        return ComplexZeros
    elif identifier == "complex_random_normal":
        return ComplexRandomNormal
    elif identifier == "complex_random_uniform":
        return ComplexRandomUniform
    else:
        raise ValueError("Invalid initializer identifier.")
