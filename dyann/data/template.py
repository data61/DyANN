from .base import BaseDataset
import numpy as np
import time
from ..util import ivecs_write, ivecs_read

class TemplateDataset(BaseDataset):
    """
    An example class providing a template for new datasets

    Usage Instructions:
        1. Copy this file and change the filename and class name for your new dataset
        2. Update ./proxy.py to include the names you have chosen
        3. Fill in each of the TODO items below (refer to existing datasets for hints if needed)
        4. Create any number of configuration sets in ../../conf/data/
            with name property set to this filename
            and scale property providing an optional parameter sweep 
    """
    
    def __init__(self, cfg):
        """
        A method for initialising internal parameters
        
        Parameters:
            cfg: OmegaConf object with current dataset properties loaded at cfg.data.X
        """
        # TODO Initialise dataset parameters here
        
    def evaluate(self, algo, cfg):
        """
        A method for evaluating ANN algorithm performance
        
        Parameters:
            algo: An ANN algorithm inheriting from BaseANN
            cfg: OmegaConf object with current evaluation properties
        Returns:
            ts: search time, update time and memory usage
            ids: topk indices returned for each query evaluated by the groundtruth
        """
        # Load queries into memory 
        vecs = self.vecs_query()

        # TODO Initialise evaluation parameters
        nq = 0 # TODO number of query samples
        ngt = 0 # TODO number of ground truth samples

        # TODO Initialise results
        ids = -1 * np.ones([ngt, cfg.topk]).astype('int') # TODO Initialise ids to -1
        ts = np.zeros([nq, 3]) # TODO search time, update time and memory usage

        # TODO Run benchmark for each query and collect results
        for query in range(nq):
            # TODO Run ANN queries through the index
            # eg. 
            # t0 = time.time()
            # ids[query,:] = algo.query(vecs=vecs[query]), topk=cfg.topk, cfg=cfg)
            # ts[query,0] = time.time() - t0
            
            # TODO Process dynamic events to update the index
            # eg.
            # t0 = time.time()
            # algo.add(vecs=vecs[query], start=query, count=self.freq)
            # ts[query,1] = time.time() - t0
            # ts[query,2] = algo.get_memory_usage(cfg.mem_type)
            pass

        # Return results
        return ts, ids

    def pregen(self, cfg):
        """
        A method for pregenerating dataset vectors and computing groundtruth data using exhaustive searches
        
        Parameters:
            cfg: OmegaConf object with current groundtruth evaluation properties
        """
        # TODO Download, generate and process any required data blobs

        # TODO Set path for groundtruth data
        gt_path = "<Path>/<Dataset>/<Parameters>_gt.ivecs"

        # Check for exisiting groundtruth files
        if not gt_path.exists():
            # TODO Initialise bruteforce algorithm
            # eg.
            # from ..algo.linear import LinearANN
            # algo = LinearANN()
            # algo.init(D=self.D(), maxN=2000*self.trunc, cfg=cfg)
            # vecs = self.vecs_base()
            # algo.add(vecs=self.vecs_base(), start=0, count=vecs.shape[0])

            # TODO Generate groundtruth
            # eg.
            # _, ids = self.evaluate(algo=algo, cfg=cfg)
            # ivecs_write(gt_path, ids)
            pass

    def vecs_train(self):
        """
        A method for loading a training set of vectors used to tune ANN algorithms that require it

        Returns:
            Training set of sample vectors as a numpy array
        """
        # TODO Load or generate and then return all training vectors
        return np.empty()

    def vecs_base(self):
        """
        A method for loading a base set of vectors used to initialise each ANN algorithm

        Returns:
            Base set of sample vectors as a numpy array
        """
        # TODO Load or generate and then return all base vectors
        return np.empty()

    def vecs_query(self):
        """
        A method for loading a query set of vectors used to evaluate each ANN algorithm

        Returns:
            Query set of sample vectors as a numpy array
        """
        # TODO Load or generate and then return all query vectors
        return np.empty()

    def groundtruth(self):
        """
        A method for loading groundtruth indices used to evaluate each ANN algorithm

        Returns:
            Groundtruth set of ANN indices as a numpy array
        """
        # TODO Set path for groundtruth data
        gt_path = "<Path>/<Dataset>/<Parameters>_gt.ivecs"
        return ivecs_read(fname=str(gt_path))
