from pathlib import Path
from math import isqrt

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")

from tqdm import tqdm

from InputPipe import InputPipe
from Discriminator import Discriminator
from Generator import Generator

class DCGAN:
    def __init__(self, sampleSize=None):
        self.mInputPipe = InputPipe(batchSize=self.mBatchSize, imageWidth=self.mImageShape[0],
                                    imageHeight=self.mImageShape[1], imageChannels=self.mImageShape[2])
        self.mInputPipe.loadAllImages(sampleSize=sampleSize)
        initWeights = ks.initializers.TruncatedNormal(stddev=0.02, mean=0)
        self.mDiscriminator = Discriminator(batchSize=self.mBatchSize,
                                            imShape=self.mImageShape, initWeights=initWeights)
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

            genLoss = self.mGenerator.loss(realOutput, fakeOutput, lossFunc=self.mLossFunc)
            discLoss = self.mDiscriminator.loss(realOutput, fakeOutput, lossFunc=self.mLossFunc)

        gradDisc = discTape.gradient(discLoss, self.mDiscriminator.mModel.trainable_variables)
        gradGen = genTape.gradient(genLoss, self.mGenerator.mModel.trainable_variables)

        self.mDiscriminator.mOptimizer.apply_gradients(zip(gradDisc, self.mDiscriminator.mModel.trainable_variables))
        self.mGenerator.mOptimizer.apply_gradients(zip(gradGen, self.mGenerator.mModel.trainable_variables))
        return (genLoss, discLoss)

    # Trains both models for x epochs
    def train(self, epochs: int, group = 4):
        # Initialise loss arrays
        self.mDiscLoss = np.zeros(epochs)
        self.mGenLoss = np.zeros(epochs)
        # Loop through each epoch

        for epoch in range(epochs):
            for chunk in self.mInputPipe.getNextImageChunk(group):
                with tqdm(total=tf.data.experimental.cardinality(chunk).numpy()) as pbar:
                    for imageBatch in chunk:
                        (genLoss, discLoss) = self.trainStep(imageBatch)
                        pbar.set_description("Progress for epoch {%s}" % epoch)
                        pbar.set_postfix_str("Generator loss: {:.5f}, Discriminator loss: {:.5f}".format(genLoss, discLoss))
                        pbar.update(1)
                        self.mDiscLoss[epoch] = discLoss
                        self.mGenLoss[epoch] = genLoss
                    pbar.close()

            if (epoch+1) % 15 == 0:
                self.save()
        return

    def genPic(self, sample=1):
        # Get random seed
        seed = tf.random.normal([sample, self.mNoiseDim])
        # Generate image with seed
        prediction = self.mGenerator.mModel(seed, training=False)
        # Unstandardisation the image
        prediction = (prediction.numpy()+1)/2
        # Calculate figure size
        nextSquare = isqrt(sample)+1
        # Get figure
        fig = plt.figure(figsize=(8, 8))
        # Add subplots
        for ind in range(1, sample+1):
            ax = fig.add_subplot(nextSquare, nextSquare, ind)
            ax.set_axis_off()
            plt.imshow(prediction[ind-1])
        plt.show()
        return

    def plotLoss(self):
        df = pd.DataFrame({
            "Epoch": np.arange(len(self.mDiscLoss)),
            "discLoss": self.mDiscLoss,
            "genLoss": self.mGenLoss
        })
        sns.lineplot(x='Epoch', y='value', hue='variable', data=pd.melt(df, ['Epoch']))
        plt.show()
        return

    mInputPipe = None
    mDiscriminator = None
    mGenerator = None
    mSaveDir = Path("TrainingCheckpoints/")
    mBatchSize = 32
    mNoiseDim = 100 # Size input to generator
    mGenLoss = None
    mDiscLoss = None
    mImageShape = (256, 256, 3)
    mLossFunc = "gan"

if __name__ == "__main__":
    gan = DCGAN(2000)
    gan.loadModels(loadCheckpoint="initial")
    gan.train(20, 10)
    gan.plotLoss()
    gan.genPic(sample=15)
    # print(gan.mDiscriminator.mModel.weights)
    gan.save()
    # gan.loadModels(loadCheckpoint="latest")
    # print(gan.mDiscriminator.mModel.weights)
    # gan.train(20)
    # gan.genPic()
    # gan.save()

