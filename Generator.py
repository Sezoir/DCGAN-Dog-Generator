from pathlib import Path

from Layers import Conv2DTransposeSN, DenseSN

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lr

class Generator:
    def __init__(self, batchSize: int = 32,
                 imShape: (int, int, int) = (64, 64, 3),
                 scale: float = 0.5,
                 initWeights: ks.initializers.Initializer = ks.initializers.RandomUniform(-1, 1)):
        self.mBatchSize = batchSize
        self.mImHeight = imShape[0]
        self.mImWidth = imShape[1]
        self.mImChannels = imShape[2]
        self.mScale = int(4 ** 1/scale)
        self.mInitWeights = initWeights
        return

    def load(self, checkpoint, dire=""):
        if checkpoint == "initial":
            self.mModel = self.createModel()
            self.setOptimizer()
        elif checkpoint == "latest":
            self.mModel = self.createModel()
            self.setOptimizer()
            checkpoint = tf.train.Checkpoint(generatorOptimizer=self.mOptimizer,
                                             generator=self.mModel)
            checkpoint.restore(tf.train.latest_checkpoint(dire / self.mSaveDir))
        else:
            # @todo: implement method for loading model at specific epochs
            ...
        return

    def save(self, dire: Path):
        checkpoint = tf.train.Checkpoint(generatorOptimizer=self.mOptimizer,
                                         generator=self.mModel)
        checkpoint.save(dire / self.mSavePre)
        return

    def createModel(self, spectralNorm: bool = True):
        model = ks.Sequential()
        if spectralNorm:
            model.add(DenseSN.DenseSN((self.mImHeight//self.mScale)*(self.mImWidth//self.mScale)*128,
                               use_bias=False, input_shape=(100,), kernel_initializer=self.mInitWeights))
            model.add(lr.BatchNormalization())
            model.add(lr.LeakyReLU())
            model.add(lr.Reshape((self.mImHeight//self.mScale, self.mImWidth//self.mScale, 128)))
            model = self.sTConvReLU(model, output=512, shape=(5, 5), stride=1)
            model.add(lr.Dropout(0.5))
            model = self.sTConvReLU(model, output=256, shape=(5, 5), stride=2)
            model.add(lr.Dropout(0.5))
            model = self.sTConvReLU(model, output=128, shape=(5, 5), stride=2)
            model = self.sTConvReLU(model, output=64, shape=(5, 5), stride=2)
            model = self.sTConvReLU(model, output=32, shape=(5, 5), stride=1)
            model.add(DenseSN.DenseSN(3, activation="tanh", kernel_initializer=self.mInitWeights))
        else:
            model.add(lr.Dense((self.mImHeight//self.mScale)*(self.mImWidth//self.mScale)*128,
                               use_bias=False, input_shape=(100,), kernel_initializer=self.mInitWeights))
            model.add(lr.BatchNormalization())
            model.add(lr.LeakyReLU())
            model.add(lr.Reshape((self.mImHeight//self.mScale, self.mImWidth//self.mScale, 128)))
            model = self.tConvReLU(model, output=512, shape=(5, 5), stride=1)
            model.add(lr.Dropout(0.5))
            model = self.tConvReLU(model, output=256, shape=(5, 5), stride=2)
            model.add(lr.Dropout(0.5))
            model = self.tConvReLU(model, output=128, shape=(5, 5), stride=2)
            model = self.tConvReLU(model, output=64, shape=(5, 5), stride=2)
            model = self.tConvReLU(model, output=32, shape=(5, 5), stride=1)
            model.add(lr.Dense(3, activation="tanh", kernel_initializer=self.mInitWeights))
        return model

    def sTConvReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                  stride: int, padding: str = "same", useBias: bool = False,
                  slope: float = 0.2) -> ks.Sequential:
        model.add(Conv2DTransposeSN.Conv2DTransposeSN(output, shape, strides=(stride, stride), padding=padding, use_bias=useBias,
                                     kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU(alpha=slope))
        return model

    def tConvReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                  stride: int, padding: str = "same", useBias: bool = False,
                  slope: float = 0.2) -> ks.Sequential:
        model.add(lr.Conv2DTranspose(output, shape, strides=(stride, stride), padding=padding,
                                     use_bias=useBias, kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU(alpha=slope))
        return model

    def setOptimizer(self, learning=0.00005, b1=0.5):
        self.mOptimizer = ks.optimizers.Adam(learning_rate=learning, beta_1=b1)
        return

    @tf.function
    def loss(self, realOutput: tf.Tensor, fakeOutput: tf.Tensor, lossFunc: str = "gan", labelSmoothing: bool = True):
        # Create labels for real and fake images
        realLabels = tf.ones_like(realOutput, dtype=tf.float32)
        fakeLabels = tf.ones_like(fakeOutput, dtype=tf.float32)
        # Apply smoothing to the labels to help stop the discriminator becoming to overconfident/underconfident about
        # its predictions. So we use the ranges [0~0.3], [0.7~1]
        if labelSmoothing:
            realLabels = realLabels - 0.3 + (np.random.random(realLabels.shape) * 0.5)
            fakeLabels = fakeLabels - 0.3 + (np.random.random(fakeLabels.shape) * 0.5)
            # fakeLabels = fakeLabels + np.random.random(fakeLabels.shape) * 0.3

        # This returns a helper function to compute the cross entropy loss
        crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Now apply the correct loss functions
        if lossFunc == "gan":
            return crossEntropy(fakeLabels, fakeOutput)
        elif lossFunc == "ralsgan":
            return (tf.reduce_mean(tf.square(realOutput - tf.reduce_mean(fakeLabels) + tf.ones_like(realLabels)))
                    + tf.reduce_mean(tf.square(fakeOutput - tf.reduce_mean(realLabels) - tf.ones_like(fakeLabels)))) / 2.

        else:
            raise ValueError("Loss function in the Generator class cannot be found.")

    def fit(self):

        return

    def evaluate(self):

        return

    def predict(self):

        return

    mBatchSize = None
    mLoad = False
    mImHeight = None
    mImWidth = None
    mImChannels = None
    mScale = None
    mInitWeights = None
    mModel: ks.Model = None
    mOptimizer = None
    mSaveDir = Path("Generator")
    mSavePre = mSaveDir / Path("ckpt")

if __name__ == "__main__":
    from InputPipe import InputPipe
    gen = Generator(imShape=(64, 64, 3),
                    initWeights=ks.initializers.TruncatedNormal(stddev=0.02, mean=0))
    gen.load()
    mod = gen.mModel
    print(mod.summary())
    print(mod.output_shape)
    # import matplotlib.pyplot as plt
    # noise = tf.random.normal([1,100])
    # generatedImage = mod(noise, training=False)
    # plt.imshow(generatedImage[0,:,:,0], cmap='gray')
    # plt.show()
    # plt.imshow(generatedImage[0, :, :, 1], cmap='gray')
    # plt.show()