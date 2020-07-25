# Python libs
from PIL import Image
from pathlib import Path
from os import listdir
from os.path import exists, abspath
# External libs
import numpy as np
import xml.etree.ElementTree as ET
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
        if not (self.mTFPath / "data.tfrecords").exists():
            self.createTFFile()
        self.loadDataset()
        return

    def formatImage(self, image):
        # Cast int8 to float
        image = tf.cast(image, dtype=tf.float32)
        # Standardisation the image
        image = (image / 127.5) - 1
        # Resize the image
        image = tf.image.resize(image, (self.mImageWidth, self.mImageHeight), antialias=True)
        return image

    def loadDataset(self):
        def readTFRecord(serialized):
            featureDescription = {
                'image': tf.io.FixedLenFeature((), tf.string),
                'height': tf.io.FixedLenFeature((), tf.int64),
                'width': tf.io.FixedLenFeature((), tf.int64),
                'channels': tf.io.FixedLenFeature((), tf.int64)
            }
            example = tf.io.parse_single_example(serialized, featureDescription)
            image = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
            imageShape = (example['height'], example['width'], example['channels'])
            image = tf.reshape(image, imageShape)
            return image
        tfDataset = tf.data.TFRecordDataset(abspath(self.mTFPath / "data.tfrecords"))
        dataset = tfDataset.map(readTFRecord)
        dataset = dataset.map(self.formatImage)
        dataset = dataset.shuffle(1000)\
                 .batch(self.mBatchSize, drop_remainder=True)\
                 .prefetch(tf.data.experimental.AUTOTUNE)

        self.mImages = dataset
        return

    def createTFFile(self):
        tf.print("Protocal files can not be found. Creating protocal files.")
        # Helper function from https://www.tensorflow.org/tutorials/load_data/tfrecord#reading_a_tfrecord_file
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        # Create helper function to serialize data
        def serialize(image, imageShape):
            feature = {
                'image': _bytes_feature(image),
                'height': _int64_feature(imageShape[0]),
                'width': _int64_feature(imageShape[1]),
                'channels': _int64_feature(imageShape[2])
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()

        with tf.io.TFRecordWriter(abspath(self.mTFPath / "data.tfrecords")) as writer:
            # Go through each breed folder
            for breedFolder in tqdm(listdir(self.mAnnPath)):
                # Go through each file in the breed folder
                for file in listdir(self.mAnnPath / breedFolder):
                    # Create path to file
                    filePath = Path(breedFolder, file)
                    # Read the image and the annotations tree
                    img = self.readImage((self.mImPath / (filePath.with_suffix(".jpg"))))
                    root = ET.parse(self.mAnnPath / filePath).getroot()
                    # Get properties for the image
                    size = root.find('size')
                    height = int(size.find('height').text)
                    width = int(size.find('width').text)
                    channels = int(size.find('depth').text)
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
                        imgByte = tf.io.serialize_tensor(imgCropped)
                        example = serialize(imgByte, (yMax-yMin, xMax-xMin, channels))
                        writer.write(example)

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
                    # imgCropped = tf.image.resize_with_pad(imgCropped, self.mImageWidth, self.mImageHeight,
                    #                                       method=interpolation, antialias=True)
                    imgCropped = tf.image.resize(imgCropped, (self.mImageWidth, self.mImageHeight),
                                                          method=interpolation, antialias=True).numpy()


                    # Store image
                    self.mImages[index] = imgCropped

                    # Increase index
                    index += 1

                    # For set sample size
                    if index == buffer-1:
                        np.random.shuffle(self.mImages)
                        return
        return

    # @todo: Move areas of function to its own function
    # @todo: Scale group depending on sampleSize
    def getNextImageChunk(self, group=2):
        size = len(self.mImages)
        chunkSize = size // group
        # Iterate through each group
        for chunk in range(group):
            # Get chunk of images
            if chunk == group:
                images = self.mImages[chunk*chunkSize:]
            else:
                images = self.mImages[chunk*chunkSize:(chunk+1)*chunkSize]
            # Change type from float64 to float32 to save memory
            images = tf.cast(images, dtype=tf.float32)
            # Create dataset
            images = tf.data.Dataset.from_tensor_slices(images).shuffle(chunkSize//2).batch(
                self.mBatchSize, drop_remainder=True)
            yield images



    mImageWidth = None
    mImageHeight = None
    mImageChannels = None
    mBatchSize = None
    mTFPath = Path("Datasets\\images")
    mAnnPath = Path("Datasets\\annotations\\Annotation")
    mImPath = Path("Datasets\\images\\Images")
    mImages = None
    mShape = None




if __name__ == "__main__":
    io = InputPipe()

    for batch in io.mImages.take(1):
        for image in batch:
            plt.imshow(image),
            plt.show()
    # imga = io.readImage("Datasets\\images\\Images\\n02085620-Chihuahua\\n02085620_199.jpg")
    # io.loadAllImages()
    # io.loadAllImages(sampleSize=100)
    #
    # for batch in io.mImages.take(1).as_numpy_iterator():
    #     for pic in batch:
    #         print(pic[32][32])
    #         plt.imshow(pic, interpolation='nearest')
    #         plt.show()

