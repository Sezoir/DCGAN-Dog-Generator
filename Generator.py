import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lr

class Generator:
    def __init__(self, batchSize: int = 32):
        self.mBatchSize = batchSize
        return

    def load(self):
        # Check whether we are loading previous model, if so load and return
        if self.mLoad == True:

            return
        # Else we create a new initial model
        initWeights = ks.initializers.TruncatedNormal(stddev=0.02, mean=0)
        self.mModel = self.createModel(initWeights)
        return

    def createModel(self, initWeights):
        model = ks.Sequential()
        model.add(lr.Dense(8*8*512, use_bias=False, input_shape=(100,)))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU())
        model.add(lr.Reshape((8, 8, 512)))
        model = self.tconvReLU(model, 256, (5, 5), 1)
        model = self.tconvReLU(model, 128, (5, 5), 2)
        model = self.tconvReLU(model, 64, (5, 5), 2)
        model = self.tconvReLU(model, 32, (5, 5), 2)
        model.add(lr.Dense(3, activation="tanh", kernel_initializer=initWeights))
        # model = self.tconvReLU(model, 1, (5, 5), 2)
        return model

    def tconvReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                  stride: int, padding="same", useBias=False,
                  slope: float = 0.2) -> ks.Sequential:
        model.add(lr.Conv2DTranspose(output, shape, strides=(stride, stride), padding=padding, use_bias=useBias))
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
    mModel = None

if __name__ == "__main__":
    from InputPipe import InputPipe
    gen = Generator()
    gen.load()
    mod = gen.mModel
    print(mod.summary())
    print(mod.output_shape)
    import matplotlib.pyplot as plt
    noise = tf.random.normal([1,100])
    generatedImage = mod(noise, training=False)
    plt.imshow(generatedImage[0,:,:,0], cmap='gray')
    plt.show()
    plt.imshow(generatedImage[0, :, :, 1], cmap='gray')
    plt.show()