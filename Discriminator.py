from pathlib import Path

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
        checkpoint.save(dire / self.mSaveDir)
        return

    def createModel(self):
        model = ks.Sequential()
        model.add(lr.Conv2D(64, (5, 5), strides=2, padding='same',
                            input_shape=[self.mImHeight, self.mImWidth, self.mImChannels],
                            kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU())
        model = self.convReLU(model, output=64, shape=(4, 4), stride=2)
        model = self.convReLU(model, output=128, shape=(4, 4), stride=2)
        model = self.convReLU(model, output=256, shape=(4, 4), stride=2)
        model.add(lr.Flatten())
        model.add(lr.Dense(1, activation='sigmoid'))
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

    def loss(self, realOutput: tf.Tensor, fakeOutput: tf.Tensor, lossFunc: str = "gan", labelSmoothing: bool = True):
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
        else:
            raise ValueError("Loss function in the Discriminator class cannot be found.")

        return realLoss + fakeLoss

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
    mSaveDir = Path("Discriminator/ckpt")

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