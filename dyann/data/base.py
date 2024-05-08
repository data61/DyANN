class BaseDataset(object):
    """ Base class for all ANN algorithms

    Inherited Methods:
        __init__: initialise internal parameters
        evaluate: evaluate an algorithms performance on the dataset
        pregen: (optional) generate any required sample vectors and groundtruth files
        vecs_train: load or generate a vector of training samples
        vecs_base: load or generate a vector of initial dataset samples
        vecs_query: load or generate a vector of query samples
        groundtruth: load groundtruth data
        D: Length of each sample vector
    """
    
    def __init__(self, cfg):
        pass

    def evaluate(self, algo, cfg):
        pass

    def pregen(self, cfg):
        pass

    def vecs_train(self):
        pass

    def vecs_base(self):
        pass

    def vecs_query(self):
        pass

    def groundtruth(self):
        pass

    def D(self):
        """Length of each sample vector"""
        vecs = self.vecs_train()
        return vecs.shape[1]
