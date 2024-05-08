from .base import BaseANN

class TemplateANN(BaseANN):
    """
    An example class providing a template for new datasets

    Usage Instructions:
        1. Copy this file and change the filename and class name for your new ANN algorithm
        2. Update ./proxy.py to include the names you have chosen
        3. Fill in each of the TODO items below (refer to existing algorithms for hints if needed)
        4. Create both the build and search configuration files in ../../conf/algo/
            with name property set to this filename
            the lists of parameters for the build and query properties will be swept
    """

    def __init__(self):
        """
        A method for initialising internal parameters
        """
        super().__init__()
        # TODO  Initialise general ANN parameters here

    def init(self, D, maxN, cfg):
        """
        A method for initialising an ANN algorithm with a dataset
        
        Parameters:
            D: The dimensionality of samples in the dataset
            maxN: The maximum number of samples required by the dataset
            cfg: OmegaConf object with current build properties
        """
        super().init(D=D, maxN=maxN, cfg=cfg)
        # TODO Initialise ANN on the dataset here

    def has_train(self):
        """
        A method for specifying if an ANN algorithm requires a training set
        
        Returns:
            A boolean value specifying if a training set is required
        """
        # TODO Set whether the ANN requires a training set
        #   if so, include an implementation of the train method inherited from BaseANN
        return False

    def do_add(self, vecs, start, count):
        """
        A method for adding samples from a dataset to an ANN algorithm
        
        Parameters:
            vecs: A matrix containing all of the samples in the dataset
            start: The index of the first sample being added to the ANN algorithm
            count: The number of samples being added to the ANN algorithm
        """
        # TODO Add each sample vector to the ANN
        #   if updating sample is possible, include an implementation of the do_update method inherited from BaseANN

    def query(self, vecs, topk, cfg):
        """
        A method for evaluating ANN algorithm performance with a dataset
        
        Parameters:
            vecs: A matrix containing all of the queries being evaluatied
            topk: The neighbourhood set size being evaluated
            cfg: OmegaConf object with current search properties

        Returns:
            An array containing the topk indices for each query evaluated
        """
        # TODO Generate ANN results for each query vector 
        return []

