from pathlib import Path
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt

from InputPipe import InputPipe
from Discriminator import Discriminator
from Generator import Generator

class DCGAN:
    def __init__(self):
        self.mInputPipe = InputPipe(batchSize=self.mBatchSize)
        self.mInputPipe.loadAllImages()#sampleSize=100
        initWeights = ks.initializers.TruncatedNormal(stddev=0.02, mean=0)
        self.mDiscriminator = Discriminator(batchSize=self.mBatchSize,
                                            imShape=self.mInputPipe.mShape, initWeights=initWeights)
        self.mGenerator = Generator(batchSize=self.mBatchSize, imShape=self.mInputPipe.mShape,
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
    def trainStep(self, images):
        noise = tf.random.normal([self.mBatchSize, self.mNoiseDim])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.mGenerator.mModel(noise, training=True)

            realOutput = self.mDiscriminator.mModel(images, training=True)
            fakeOutput = self.mDiscriminator.mModel(generatedImages, training=True)

            genLoss = self.mGenerator.loss(realOutput, fakeOutput)
            discLoss = self.mDiscriminator.loss(realOutput, fakeOutput)

        gradGen = genTape.gradient(genLoss, self.mGenerator.mModel.trainable_variables)
        gradDisc = discTape.gradient(discLoss, self.mDiscriminator.mModel.trainable_variables)

        self.mGenerator.mOptimizer.apply_gradients(zip(gradGen, self.mGenerator.mModel.trainable_variables))
        self.mDiscriminator.mOptimizer.apply_gradients(zip(gradDisc, self.mDiscriminator.mModel.trainable_variables))
        return

    # Trains both models for x epochs
    def train(self, epochs: int):
        for epoch in range(epochs):
            start = time.time()

            for imageBatch in self.mInputPipe.mImages:
                self.trainStep(imageBatch)

            if (epoch+1) % 15 == 0:
                self.save()

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        self.genPic()
        return

    def genPic(self):
        # Get random seed
        seed = tf.random.normal([1, self.mNoiseDim])
        # Generate image with seed
        prediction = self.mGenerator.mModel(seed, training=False)
        # Unstandardisation the image
        prediction = (prediction+1)*127.5
        # Show image
        plt.imshow(prediction.numpy()[0])
        plt.show()
        return




    mInputPipe = None
    mDiscriminator = None
    mGenerator = None
    mSaveDir = Path("TrainingCheckpoints/")
    mBatchSize = 32
    mNoiseDim = 100 # Size input to generator

if __name__ == "__main__":
    gan = DCGAN()
    gan.loadModels(loadCheckpoint="initial")
    # print(gan.mDiscriminator.mModel.weights)
    # gan.save()
    # gan.loadModels(loadCheckpoint="latest")
    # print(gan.mDiscriminator.mModel.weights)
    gan.train(50)
