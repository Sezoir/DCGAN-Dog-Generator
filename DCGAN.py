import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from InputPipe import InputPipe
from Discriminator import Discriminator
from Generator import Generator

class DCGAN:
    def __init__(self):
        self.mInputPipe = InputPipe()
        self.mInputPipe.loadAllImages()
        initWeights = ks.initializers.TruncatedNormal(stddev=0.02, mean=0)
        self.mDiscriminator = Discriminator(self.mInputPipe.mImages, batchSize=self.mInputPipe.mBatchSize,
                                            imShape=self.mInputPipe.mShape, initWeights=initWeights)
        self.mGenerator = Generator(batchSize=self.mInputPipe.mBatchSize, imShape=self.mInputPipe.mShape,
                                    initWeights=initWeights)
        return

    def loadModels(self):
        self.mDiscriminator.load()
        self.mGenerator.load()
        return




    mInputPipe = None
    mDiscriminator = None
    mGenerator = None


if __name__ == "__main__":
    gan = DCGAN()
    gan.loadModels()
