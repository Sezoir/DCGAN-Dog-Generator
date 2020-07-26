import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as lr
import tensorflow.keras as ks
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops, math_ops

class DenseSN(lr.Dense):
    def build(self, input_shape):
        kernalShape = (input_shape[-1], self.units)
        self.kernel = self.add_weight(shape=kernalShape,
                                      initializer=self.kernel_initializer,
                                      name='kernal',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=ks.initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

        self.input_spec = lr.InputSpec(min_ndim=2, axes={-1: input_shape[-1]})

        self.build = True
        return

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        wBar = self._computeWeights(training)
        # Get output
        output = math_ops.matmul(inputs, wBar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def _computeWeights(self, training: bool):
        # L2 Norm
        def l2Norm(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        # Power iteration
        def powIt(W, u):
            _u = u
            _v = l2Norm(K.dot(_u, K.transpose(W)))
            _u = l2Norm(K.dot(_v, W))
            return _u, _v
        # Get kernal shape and flatten
        wShape = self.kernel.shape.as_list()
        wReshaped = K.reshape(self.kernel, [-1, wShape[-1]])
        # Power iteration
        _u, _v = powIt(wReshaped, tf.identity(self.u))
        # Calculate sigma
        sigma = K.dot(_v, wReshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # Caculate Wsn
        wBar = wReshaped / sigma
        # Reshape tensor
        wBar = K.reshape(wBar, wShape)
        # Update u
        if training:
            self.u.assign(_u)
        return wBar