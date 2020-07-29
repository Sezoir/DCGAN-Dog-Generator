from pathlib import Path

from Layers import Conv2DSN, DenseSN

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lr

class Discriminator:
    def __init__(self,
                 batchSize: int = 32,
                 imShape: (int, int, int) = (64, 64, 3),
                 initWeights: ks.initializers.Initializer = ks.initializers.RandomUniform(-1, 1)):
        self.mBatchSize = batchSize
        self.mImHeight = imShape[0]
        self.mImWidth = imShape[1]
        self.mImChannels = imShape[2]
        self.mInitWeights = initWeights
        return

    def load(self, checkpoint, dire=""):
        if checkpoint == "initial":
            self.mModel = self.createModel()
            self.setOptimizer()
        elif checkpoint == "latest":
            self.mModel = self.createModel()
            self.setOptimizer()
            checkpoint = tf.train.Checkpoint(discriminatorOptimizer=self.mOptimizer,
                                             discriminator=self.mModel)
            checkpoint.restore(tf.train.latest_checkpoint(dire / self.mSaveDir))
        else:
            # @todo: implement method for loading model at specific epochs
            ...
        return

    def save(self, dire: Path):
        checkpoint = tf.train.Checkpoint(discriminatorOptimizer=self.mOptimizer,
                                         discriminator=self.mModel)
        checkpoint.save(dire / self.mSavePre)
        return

    def createModel(self, spectralNorm: bool = True):
        model = ks.Sequential()
        if spectralNorm:
            model.add(Conv2DSN.Conv2DSN(64, (5, 5), strides=2, padding='same',
                               input_shape=[self.mImHeight, self.mImWidth, self.mImChannels],
                               kernel_initializer=self.mInitWeights))
            # model.add(lr.BatchNormalization())
            model.add(lr.LeakyReLU(alpha=0.2))
            model = self.sConvReLU(model, output=64, shape=(4, 4), stride=2)
            model = self.sConvReLU(model, output=128, shape=(4, 4), stride=2)
            model = self.sConvReLU(model, output=256, shape=(4, 4), stride=2)
            model.add(lr.Flatten())
            model.add(DenseSN.DenseSN(1, activation='sigmoid'))
        else:
            model.add(lr.Conv2D(64, (5, 5), strides=2, padding='same',
                                input_shape=[self.mImHeight, self.mImWidth, self.mImChannels],
                                kernel_initializer=self.mInitWeights))
            model.add(lr.BatchNormalization())
            model.add(lr.LeakyReLU(alpha=0.2))
            model = self.convReLU(model, output=64, shape=(4, 4), stride=2)
            model = self.convReLU(model, output=128, shape=(4, 4), stride=2)
            model = self.convReLU(model, output=256, shape=(4, 4), stride=2)
            model.add(lr.Flatten())
            model.add(lr.Dense(1, activation='sigmoid'))
        return model

    def sConvReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                 stride: int, padding: str = "same", useBias: bool = False,
                 slope: float = 0.2) -> ks.Sequential:
        model.add(Conv2DSN.Conv2DSN(output, shape, strides=(stride, stride), padding=padding, use_bias=useBias,
                            kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU(alpha=slope))
        return model

    def convReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                 stride: int, padding: str = "same", useBias: bool = False,
                 slope: float = 0.2) -> ks.Sequential:
        model.add(lr.Conv2D(output, shape, strides=(stride, stride), padding=padding, use_bias=useBias,
                            kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU(alpha=slope))
        return model

    def setOptimizer(self, learning = 0.0002, b1=0.5):
        self.mOptimizer = ks.optimizers.Adam(learning_rate=learning, beta_1=b1)
        return

    @tf.function
    def loss(self, realOutput: tf.Tensor, fakeOutput: tf.Tensor,
             lossFunc: str = "gan", labelSmoothing: bool = True,
             noise: bool = True):
        def addGaussianNoise(image):
            noise = tf.random.normal(shape=realOutput.shape, mean=0, stddev=50/255, dtype=tf.float32)
            image = image+noise
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image
        # Add noise to images
        if noise:
            tf.map_fn(addGaussianNoise, realOutput, parallel_iterations=10)
            tf.map_fn(addGaussianNoise, fakeOutput, parallel_iterations=10)
        # Create labels for real and fake images
        realLabels = tf.ones_like(realOutput)
        fakeLabels = tf.zeros_like(fakeOutput)
        # Apply smoothing to the labels to help stop the discriminator becoming to overconfident/underconfident about
        # its predictions. So we use the ranges [0~0.3], [0.7~1]
        if labelSmoothing:
            realLabels = realLabels - 0.3 + (np.random.random(realLabels.shape)*0.5)
            fakeLabels = fakeLabels + np.random.random(fakeLabels.shape)*0.3

        # This returns a helper function to compute the cross entropy loss
        crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Now apply the correct loss functions
        if lossFunc == "gan":
            realLoss = crossEntropy(realLabels, realOutput)
            fakeLoss = crossEntropy(fakeLabels, fakeOutput)
            return realLoss + fakeLoss
        elif lossFunc == "ralsgan":
            return (tf.reduce_mean(tf.square(realOutput - tf.reduce_mean(fakeLabels) - tf.ones_like(realLabels)))
                    + tf.reduce_mean(tf.square(fakeOutput - tf.reduce_mean(realLabels) + tf.ones_like(fakeLabels)))) / 2.
        else:
            raise ValueError("Loss function in the Discriminator class cannot be found.")

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
    mInitWeights = None
    mModel: ks.Model = None
    mOptimizer = None
    mSaveDir = Path("Discriminator")
    mSavePre = mSaveDir / Path("ckpt")

if __name__ == "__main__":
    # from InputPipe import InputPipe
    # io = InputPipe()
    # io.loadAllImages()
    disc = Discriminator(imShape=(64, 64, 3),
                         initWeights=ks.initializers.TruncatedNormal(stddev=0.02, mean=0))
    disc.load(checkpoint="initial")
    mod = disc.mModel
    mod.summary()
    print(mod.output_shape)