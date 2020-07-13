import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lr

class Generator:
    def __init__(self, batchSize: int = 32,
                 imShape: (int, int, int) = (64, 64, 3),
                 scale: float = 0.5,
                 load: bool = False,
                 initWeights: ks.initializers.Initializer = ks.initializers.RandomUniform(-1, 1)):
        self.mBatchSize = batchSize
        self.mImHeight = imShape[0]
        self.mImWidth = imShape[1]
        self.mImChannels = imShape[2]
        self.mScale = int(4 ** 1/scale)
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
        model.add(lr.Dense((self.mImHeight//self.mScale)*(self.mImWidth//self.mScale)*512,
                           use_bias=False, input_shape=(100,), kernel_initializer=self.mInitWeights))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU())
        model.add(lr.Reshape((self.mImHeight//self.mScale, self.mImWidth//self.mScale, 512)))
        model = self.tconvReLU(model, output=256, shape=(5, 5), stride=1)
        model = self.tconvReLU(model, output=128, shape=(5, 5), stride=2)
        model = self.tconvReLU(model, output=64, shape=(5, 5), stride=2)
        model = self.tconvReLU(model, output=32, shape=(5, 5), stride=2)
        model.add(lr.Dense(3, activation="tanh", kernel_initializer=self.mInitWeights))
        return model

    def tconvReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                  stride: int, padding: str = "same", useBias: bool = False,
                  slope: float = 0.2) -> ks.Sequential:
        model.add(lr.Conv2DTranspose(output, shape, strides=(stride, stride), padding=padding, use_bias=useBias,
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

    mBatchSize = None
    mLoad = False
    mImHeight = None
    mImWidth = None
    mImChannels = None
    mScale = None
    mInitWeights = None
    mModel = None

if __name__ == "__main__":
    from InputPipe import InputPipe
    gen = Generator(imShape=(64, 64, 3),
                    load=False,
                    initWeights=ks.initializers.TruncatedNormal(stddev=0.02, mean=0))
    gen.load()
    mod = gen.mModel
    print(mod.summary())
    print(mod.output_shape)
    # import matplotlib.pyplot as plt
    # noise = tf.random.normal([1,100])
    # generatedImage = mod(noise, training=False)
    # plt.imshow(generatedImage[0,:,:,0], cmap='gray')
    # plt.show()
    # plt.imshow(generatedImage[0, :, :, 1], cmap='gray')
    # plt.show()