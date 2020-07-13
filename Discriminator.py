import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lr

class Discriminator:
    def __init__(self, dataset: tf.data.Dataset,
                 batchSize: int = 32,
                 imShape: (int, int, int) = (64, 64, 3),
                 load: bool = False,
                 initWeights: ks.initializers.Initializer = ks.initializers.RandomUniform(-1, 1)):
        self.mDataset = dataset
        self.mBatchSize = batchSize
        self.mImHeight = imShape[0]
        self.mImWidth = imShape[1]
        self.mImChannels = imShape[2]
        self.mLoad = load
        self.mInitWeights = initWeights
        return

    def load(self):
        # Check whether we are loading previous model, if so load and return
        if self.mLoad == True:

            return
        # Else we create a new initial model
        self.mModel = self.createModel()
        return

    def createModel(self):
        model = ks.Sequential()
        model.add(lr.Conv2D(64, (5, 5), strides=2, padding='same',
                            input_shape=[self.mImHeight, self.mImWidth, 3], kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU())
        model = self.convReLU(model, output=64, shape=(4, 4), stride=2)
        model = self.convReLU(model, output=128, shape=(4, 4), stride=2)
        model = self.convReLU(model, output=256, shape=(4, 4), stride=2)
        model.add(lr.Flatten())
        model.add(lr.Dense(1, activation='sigmoid'))
        return model

    def convReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                 stride: int, padding="same", useBias: bool = False,
                 slope: float = 0.2) -> ks.Sequential:
        model.add(lr.Conv2D(output, shape, strides=(stride, stride), padding=padding, use_bias=useBias,
                            kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU(alpha=slope))
        return model

    def fit(self):

        return

    def evaluate(self):

        return

    def save(self):

        return

    def predict(self):

        return

    mDataset = None
    mBatchSize = None
    mLoad = False
    mImHeight = None
    mImWidth = None
    mImChannels = None
    mInitWeights = None
    mModel = None

if __name__ == "__main__":
    # from InputPipe import InputPipe
    # io = InputPipe()
    # io.loadAllImages()
    disc = Discriminator(tf.data.Dataset.from_tensor_slices([1,2,3]),
                         imShape=(64, 64, 3),
                         load=False,
                         initWeights=ks.initializers.TruncatedNormal(stddev=0.02, mean=0))
    mod = disc.createModel()
    mod.summary()
    print(mod.output_shape)