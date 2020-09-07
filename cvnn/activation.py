import tensorflow as tf
from tensorflow.python.keras import backend as K


class ComplexReLU(tf.keras.layers.Layer):
    """Complex rectified linear unit.

      With default values, it returns element-wise `max(x, 0)` separately for real and imaginary part.

      Otherwise, it follows:
      `f(x) = max_value` for `x >= max_value`,
      `f(x) = x` for `threshold <= x < max_value`,
      `f(x) = alpha * (x - threshold)` otherwise.
    """
    def __init__(self,
                 alpha_real=0.,
                 alpha_imag=0.,
                 max_value_real=None,
                 max_value_imag=None,
                 threshold_real=0,
                 threshold_imag=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha_real = alpha_real
        self.alpha_imag = alpha_imag
        self.max_value_real = max_value_real
        self.max_value_imag = max_value_imag
        self.threshold_real = threshold_real
        self.threshold_imag = threshold_imag

    def call(self, inputs, **kwargs):
        inputs_real, inputs_imag = tf.math.real(inputs), tf.math.imag(inputs)
        real = K.relu(inputs_real,
                      max_value=self.max_value_real,
                      threshold=self.threshold_real,
                      alpha=self.alpha_real)
        imag = K.relu(inputs_imag,
                      max_value=self.max_value_imag,
                      threshold=self.threshold_imag,
                      alpha=self.alpha_imag)
        return tf.complex(real, imag)


def get(identifier):
    if identifier == 'linear':
        return lambda x: x
    elif identifier == 'relu':
        return ComplexReLU
    else:
        raise ValueError(f"Invalid name for activation function. "
                         f"Given '{identifier}', but expected 'linear' or 'relu'.")
