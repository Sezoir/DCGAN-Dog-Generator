import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lr

class Discriminator:
    def __init__(self, dataset: tf.data.Dataset,
                 batchSize: int = 32, imageSize: (int, int, int) = (64, 64, 3)):
        self.mDataset = dataset
        self.mBatchSize = batchSize
        return

    def load(self):

        return

    def createModel(self):
        model = ks.Sequential()
        model.add(lr.Conv2D(64, (5, 5), strides=2, padding='same', input_shape=[64, 64, 3]))
        model.add(lr.BatchNormalization())
        model.add(lr.LeakyReLU())
        model = self.convReLU(model, 64, (4, 4), 2)
        model = self.convReLU(model, 128, (4, 4), 2)
        model = self.convReLU(model, 256, (4, 4), 2)
        model.add(lr.Flatten())
        model.add(lr.Dense(1, activation='sigmoid'))
        return model

    def convReLU(self, model: ks.Sequential, output: int, shape: (int, int),
                  stride: int, padding="same", useBias=False,
                  slope: float = 0.2) -> ks.Sequential:
        model.add(lr.Conv2D(output, shape, strides=(stride, stride), padding=padding, use_bias=useBias))
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

if __name__ == "__main__":
    # from InputPipe import InputPipe
    # io = InputPipe()
    # io.loadAllImages()
    disc = Discriminator(4)
    mod = disc.createModel()
    mod.summary()
    print(mod.output_shape)