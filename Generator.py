

class Generator:
    def __init__(self, dataset, batchSize=32):
        self.mDataset = dataset
        self.mBatchSize = batchSize
        return

    def load(self):

        return

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
    from InputPipe import InputPipe
    io = InputPipe()
    io.loadAllImages()
    gen = Generator(io.mImages)
    gen.load()