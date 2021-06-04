import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils

from cvnn import initializers, utils


class ComplexDense(layers.Dense):
    """ Complex-valued fully-connected layer. """

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='complex_random_normal',
                 bias_initializer='complex_zeros',
                 dtype=tf.dtypes.complex64,
                 **kwargs):
        """

        Complex-valued fully-connected layer.

        Parameters
        ----------
            units: int,
                Layer output dimension.
            use_bias: bool, optional (default=True)
                Bias usage in layer building.
            kernel_initializer: str, optional (default='complex_random_normal')
                Type of kernel initializer.
            bias_initializer: str, optional (default='complex_zeros')
                Type of bias initializer.
            dtype:  tf.dtypes.Dtype, optional (default=tf.dtypes.complex64)
                Data type of kernel and bias.
        """
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
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)


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
        return tf.complex(real_output, imag_output)


class ComplexConv2D(tf.keras.models.Model):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(ComplexConv2D, self).__init__()
        self.conv_re = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding,
                                              data_format, dilation_rate, groups, activation,
                                              use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
                                              bias_regularizer, activity_regularizer, kernel_constraint,
                                              bias_constraint, **kwargs)
        self.conv_im = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding,
                                              data_format, dilation_rate, groups, activation,
                                              use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
                                              bias_regularizer, activity_regularizer, kernel_constraint,
                                              bias_constraint, **kwargs)

    def build(self, input_shape):
        self.conv_re.build(input_shape)
        self.conv_im.build(input_shape)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        inputs_re, inputs_im = tf.math.real(inputs), tf.math.imag(inputs)
        h_rr = self.conv_re(inputs_re, training=training)
        h_ii = self.conv_im(inputs_im, training=training)
        h_ir = self.conv_im(inputs_re, training=training)
        h_ri = self.conv_re(inputs_im, training=training)
        return tf.complex(h_rr - h_ii, h_ir + h_ri)

    def get_config(self):
        return self.conv_re.get_config()
