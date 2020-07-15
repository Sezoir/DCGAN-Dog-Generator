from pathlib import Path
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from tqdm import tqdm

from InputPipe import InputPipe
from Discriminator import Discriminator
from Generator import Generator

class DCGAN:
    def __init__(self, sampleSize=None):
        self.mInputPipe = InputPipe(batchSize=self.mBatchSize)
        self.mInputPipe.loadAllImages(sampleSize=sampleSize)
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
        return (genLoss, discLoss)

    # Trains both models for x epochs
    def train(self, epochs: int):
        for epoch in range(epochs):
            # start = time.time()

            with tqdm(total=tf.data.experimental.cardinality(self.mInputPipe.mImages).numpy()) as pbar:
                for imageBatch in self.mInputPipe.mImages:
                    (genLoss, discLoss) = self.trainStep(imageBatch)
                    pbar.set_description("Progress for epoch {%s}" % epoch)
                    pbar.set_postfix_str("Generator loss: {:.5f}, Discriminator loss: {:.5f}".format(genLoss, discLoss))
                    # pbar.set_description("Generator loss: %s, Discriminator loss: %s, Progress for epoch {%s}" % genLoss, discLoss, epoch)
                    pbar.update(1)
                pbar.close()

            if (epoch+1) % 15 == 0:
                self.save()
        return

    def genPic(self):
        # Get random seed
        seed = tf.random.normal([1, self.mNoiseDim])
        # Generate image with seed
        prediction = self.mGenerator.mModel(seed, training=False)
        # Unstandardisation the image
        prediction = (prediction.numpy()[0]+1)/2
        # Show image
        plt.imshow(prediction)
        plt.show()
        return




    mInputPipe = None
    mDiscriminator = None
    mGenerator = None
    mSaveDir = Path("TrainingCheckpoints/")
    mBatchSize = 64
    mNoiseDim = 100 # Size input to generator

if __name__ == "__main__":
    gan = DCGAN(100)
    gan.loadModels(loadCheckpoint="latest")
    gan.genPic()
    # print(gan.mDiscriminator.mModel.weights)
    # gan.save()
    # gan.loadModels(loadCheckpoint="latest")
    # print(gan.mDiscriminator.mModel.weights)
    gan.train(10)
    gan.genPic()
    gan.save()

