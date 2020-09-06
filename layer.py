from tensorflow import dtypes
from tensorflow import complex
from tensorflow.keras import layers
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils

import utils
import initializer


class ComplexDense(layers.Dense):
    """ Complex-valued fully-connected layer. """

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='complex_random_normal',
                 bias_initializer='complex_zeros',
                 dtype=dtypes.complex64,
                 **kwargs):
        utils.check_complex_dtype(dtype)
        kwargs.update({'dtype': dtype})
        super(ComplexDense, self).__init__(
            units=units,
            use_bias=use_bias,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs)
        self.kernel_initializer = initializer.get(kernel_initializer)
        self.bias_initializer = initializer.get(bias_initializer)


class ComplexDropout(layers.Dropout):
    """ Dropout for complex-valued layer. """

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        real_inputs = K.math_ops.real(inputs)
        imag_inputs = K.math_ops.imag(inputs)

        def dropped_inputs(input_type):
            def _dropped_inputs():
                if input_type == 'real':
                    _inputs = real_inputs
                elif input_type == 'imag':
                    _inputs = imag_inputs
                else:
                    raise ValueError("Invalid input type. "
                                     "Available values are 'real' and 'imag'")
                return nn.dropout(
                    _inputs,
                    noise_shape=self._get_noise_shape(_inputs),
                    seed=self.seed,
                    rate=self.rate)

            return _dropped_inputs

        real_output = tf_utils.smart_cond(training,
                                          dropped_inputs('real'),
                                          lambda: array_ops.identity(real_inputs))
        imag_output = tf_utils.smart_cond(training,
                                          dropped_inputs('imag'),
                                          lambda: array_ops.identity(imag_inputs))
        return complex(real_output, imag_output)
