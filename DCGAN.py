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
    def train(self, epochs: int):
        # Initialise loss arrays
        self.mDiscLoss = np.zeros(epochs)
        self.mGenLoss = np.zeros(epochs)
        # Loop through each epoch

        for epoch in range(epochs):
            ind = 1
            with tqdm(total=self.mInputPipe.getImgCount()//self.mBatchSize) as pbar:
                for imageBatch in self.mInputPipe.mImages:
                    (genLoss, discLoss) = self.trainStep(imageBatch)
                    self.mDiscLoss[epoch] += discLoss
                    self.mGenLoss[epoch] += genLoss
                    pbar.set_description("Progress for epoch {%s}" % epoch)
                    pbar.set_postfix_str("Generator loss: {:.5f}, Discriminator loss: {:.5f}".format(self.mGenLoss[epoch]/ind, self.mDiscLoss[epoch]/ind))
                    pbar.update(1)
                    ind += 1
                pbar.close()

            if (epoch+1) % 5 == 0:
                self.save()

            if (epoch+1) % 20 == 0:
                self.plotLoss()
                self.genPic(sample=25)
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
    mBatchSize = 64
    mNoiseDim = 100 # Size input to generator
    mGenLoss = None
    mDiscLoss = None
    mImageShape = (128, 128, 3)
    mLossFunc = "gan"

if __name__ == "__main__":
    gan = DCGAN()
    gan.loadModels(loadCheckpoint="latest")
    gan.train(100)
    gan.plotLoss()
    gan.genPic(sample=25)
    # print(gan.mDiscriminator.mModel.weights)
    gan.save()
    # gan.loadModels(loadCheckpoint="latest")
    # print(gan.mDiscriminator.mModel.weights)
    # gan.train(20)
    # gan.genPic()
    # gan.save()

