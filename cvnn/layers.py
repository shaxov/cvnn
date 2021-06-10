import six
import abc
import functools

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops import nn_ops

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


class ComplexConv(Conv):

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='complex_random_normal',
                 bias_initializer='complex_zeros',
                 trainable=True,
                 name=None,
                 conv_op=None,
                 **kwargs):
        super(ComplexConv, self).__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            trainable=trainable,
            dtype='complex64',
            name=name,
            conv_op=conv_op,
            **kwargs,
        )

    def call(self, inputs):
        inputs_re, inputs_im = tf.math.real(inputs), tf.math.imag(inputs)
        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs_re = array_ops.pad(inputs_re, self._compute_causal_padding(inputs_re))
            inputs_im = array_ops.pad(inputs_im, self._compute_causal_padding(inputs_im))

        kernel_re, kernel_im = tf.math.real(self.kernel), tf.math.imag(self.kernel)

        outputs_rr = self._convolution_op(inputs_re, kernel_re)
        outputs_ii = self._convolution_op(inputs_im, kernel_im)
        outputs_ir = self._convolution_op(inputs_re, kernel_im)
        outputs_ri = self._convolution_op(inputs_im, kernel_re)
        outputs = tf.complex(outputs_rr - outputs_ii, outputs_ir + outputs_ri)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = nn_ops.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class ComplexConv1D(ComplexConv):

    def __init__(self, filters, kernel_size, strides=1, padding='valid',
                 data_format=None, dilation_rate=1, groups=1, activation=None,
                 use_bias=True, kernel_initializer='complex_random_normal',
                 bias_initializer='complex_zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(ComplexConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class ComplexConv2D(ComplexConv):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
                 use_bias=True, kernel_initializer='complex_random_normal',
                 bias_initializer='complex_zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(ComplexConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class ComplexConv3D(ComplexConv):

    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding='valid',
                 data_format=None, dilation_rate=(1, 1, 1), groups=1, activation=None,
                 use_bias=True, kernel_initializer='complex_random_normal',
                 bias_initializer='complex_zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(ComplexConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


class ProdLayer1D(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[..., None]


class ProdLayer2D(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        inputs2d = inputs[:, :, None] * inputs[:, None, :]
        return inputs2d[..., None]


class ProdLayer3D(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        inputs3d = inputs[:, :, None, None] * inputs[:, None, :, None] * inputs[:, None, None, :]
        return inputs3d[..., None]
