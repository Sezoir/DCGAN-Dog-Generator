from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from os import listdir
import tensorflow as tf
from tqdm import tqdm


import matplotlib.pyplot as plt

class InputPipe():

    def __init__(self, batchSize=32, imageWidth=64, imageHeight=64, imageChannels=3):
        self.mBatchSize = batchSize
        self.mImageWidth = imageWidth
        self.mImageHeight = imageHeight
        self.mImageChannels = imageChannels
        self.mShape = (imageHeight, imageWidth, imageChannels)
        return

    def readImage(self, path) -> np.ndarray:
        # Attempt to load file
        try:
            img = Image.open(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"From InputPipe::readImage: No such file or directory: '{path}'")
        # Convert to RGB image
        img = img.convert('RGB')
        # Return image as a numpy array
        return np.array(img)

    def getImgCount(self) -> int:
        counter = 0
        # Go through each breed folder
        for breedFolder in listdir(self.mAnnPath):
            # Go through each file in the breed folder
            for file in listdir(self.mAnnPath / breedFolder):
                # Create path to file
                filePath = Path(breedFolder, file)
                # Read the image and the annotations tree
                objects = ET.parse(self.mAnnPath / filePath).getroot().findall('object')
                for object in objects:
                    counter += 1
        return counter

    def loadAllImages(self, sampleSize = None):
        # Get count of images
        if sampleSize == None:
            buffer = self.getImgCount()
        else:
            buffer = sampleSize
        # Initialise empty numpy array
        self.mImages = np.zeros((buffer, self.mImageWidth, self.mImageHeight, self.mImageChannels))
        # Create index
        index = 0
        # Go through each breed folder
        for breedFolder in listdir(self.mAnnPath):
            # Go through each file in the breed folder
            for file in tqdm(listdir(self.mAnnPath / breedFolder)):
                # Create path to file
                filePath = Path(breedFolder, file)
                # Read the image and the annotations tree
                img = self.readImage((self.mImPath / (filePath.with_suffix(".jpg"))))
                root = ET.parse(self.mAnnPath / filePath).getroot()
                # Get properties for the image
                size = root.find('size')
                height = int(size.find('height').text)
                width = int(size.find('width').text)
                # channels = size.find('depth')
                objects = root.findall('object')
                # Pictures may have multiple dogs in them
                # So we iterate through each dog in the picture and crop the image to the specific dog
                for object in objects:
                    # Get the margins for the dog
                    bndBox = object.find('bndbox')
                    xMin = int(bndBox.find('xmin').text)
                    xMax = int(bndBox.find('xmax').text)
                    yMin = int(bndBox.find('ymin').text)
                    yMax = int(bndBox.find('ymax').text)
                    # Make the margins slightly bigger
                    xMin = max(0, xMin - 4)
                    xMax = min(width, xMax + 4)
                    yMin = max(0, yMin - 4)
                    yMax = min(height, yMax + 4)
                    # Crop the image to focus on the specific dog
                    # @todo: test whether adding a wider margin helps the discriminator more
                    imgCropped = img[yMin:yMax, xMin:xMax, :]

                    # Standardisation the images
                    imgCropped = (imgCropped / 127.5) - 1

                    # Get the interpolation method for scaling the picture
                    if xMax - xMin > width:
                        interpolation = tf.image.ResizeMethod.AREA  # Shrink
                    else:
                        interpolation = tf.image.ResizeMethod.BICUBIC   # Stretch

                    # Resize image with pad to reserve aspect ratio
                    imgCropped = tf.image.resize_with_pad(imgCropped, self.mImageWidth, self.mImageHeight,
                                                          method=interpolation, antialias=True)

                    # Store image
                    self.mImages[index] = imgCropped

                    # Increase index
                    index += 1

                    if index == buffer-1:
                        # Change type from float64 to float32 to save memory
                        self.mImages = tf.cast(self.mImages, dtype=tf.float32)
                        # Create dataset
                        self.mImages = tf.data.Dataset.from_tensor_slices(self.mImages).shuffle(buffer).batch(
                            self.mBatchSize, drop_remainder=True)
                        return
        return


    mImageWidth = None
    mImageHeight = None
    mImageChannels = None
    mBatchSize = None
    mAnnPath = Path("Datasets\\annotations\\Annotation")
    mImPath = Path("Datasets\\images\\Images")
    mImages = None
    mShape = None




if __name__ == "__main__":
    io = InputPipe()
    # imga = io.readImage("Datasets\\images\\Images\\n02085620-Chihuahua\\n02085620_199.jpg")
    io.loadAllImages(sampleSize=100)

    for batch in io.mImages.take(1).as_numpy_iterator():
        for pic in batch:
            print(pic[32][32])
            plt.imshow(pic, interpolation='nearest')
            plt.show()

