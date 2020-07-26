
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as lr
import tensorflow.keras as ks
import tensorflow.keras.backend as K

class Conv2DSN(lr.Conv2D):

    def build(self, input_shape):

        kernalShape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(shape=kernalShape,
                                      initializer=self.kernel_initializer,
                                      name='kernal',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
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

        self.input_spec = lr.InputSpec(ndim=self.rank+2,
                                       axes={-1: input_shape[-1]})
        self.build = True
        return

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        self.kernel.assign(self._computeWeights(training))

        output = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        if self.use_bias:
            output = K.bias_add(output,
                                self.bias,
                                data_format=self.data_format)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def _computeWeights(self, training: bool):
        # L2 Norm
        def l2Norm(v, eps=1e-12):
            return v / (K.sum(v**2)**0.5 + eps)
        # Power iteration
        def powIt(W, u):
            _u = u
            _v = l2Norm(K.dot(_u, K.transpose(W)))
            _u = l2Norm(K.dot(_v, W))
            return _u, _v
        # Get kernal shape and flatten
        wShape = self.kernel.shape.as_list()
        wReshaped = K.reshape(self.kernel, [-1, wShape[-1]])
        # Get the power iteration
        _u, _v = powIt(wReshaped, tf.identity(self.u))
        # Calculate sigma
        sigma = K.dot(_v, wReshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        # Calculate Wsn with sigma
        wBar = wReshaped / sigma
        # Reshape weight tensor
        wBar = K.reshape(wBar, wShape)
        # Assign/update _u
        if training:
            self.u.assign(_u)
        return wBar

if __name__ == "__main__":
    layer = Conv2DSN(64, (4, 4))
    layer.build((16,16,3))
    # layer.call([15,15,3])
    layer._computeWeights(False)