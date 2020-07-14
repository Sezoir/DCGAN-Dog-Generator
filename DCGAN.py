from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from InputPipe import InputPipe
from Discriminator import Discriminator
from Generator import Generator

class DCGAN:
    def __init__(self):
        self.mInputPipe = InputPipe()
        self.mInputPipe.loadAllImages(sampleSize=100)
        initWeights = ks.initializers.TruncatedNormal(stddev=0.02, mean=0)
        self.mDiscriminator = Discriminator(self.mInputPipe.mImages, batchSize=self.mInputPipe.mBatchSize,
                                            imShape=self.mInputPipe.mShape, initWeights=initWeights)
        self.mGenerator = Generator(batchSize=self.mInputPipe.mBatchSize, imShape=self.mInputPipe.mShape,
                                    initWeights=initWeights)
        return

    def loadModels(self, loadCheckpoint):
        self.mDiscriminator.load(checkpoint=loadCheckpoint, dire=self.mSaveDir)
        self.mGenerator.load(checkpoint=loadCheckpoint, dire=self.mSaveDir)
        return

    def save(self):
        self.mGenerator.save(self.mSaveDir)
        self.mDiscriminator.save(self.mSaveDir)
        return

    # The training step for both models for 1 epoch
    @tf.function
    def trainStep(self):

        return

    # Trains both models for x epochs
    def train(self):

        return




    mInputPipe = None
    mDiscriminator = None
    mGenerator = None
    mSaveDir = Path("TrainingCheckpoints/")
    mCheckPoint = None

if __name__ == "__main__":
    gan = DCGAN()
    gan.loadModels(loadCheckpoint="initial")
    print(gan.mDiscriminator.mModel.weights)
    gan.save()
    gan.loadModels(loadCheckpoint="latest")
    print(gan.mDiscriminator.mModel.weights)
