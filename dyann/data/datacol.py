from .base import BaseDataset
from pathlib import Path
from urllib.request import urlretrieve
import tarfile
import numpy as np
import time
from ..util import ivecs_read, fvecs_read, ivecs_write, lerp

class OnlineDataCollection(BaseDataset):
    """
    A class for simulation of online data collection datasets

    Data source: http://corpus-texmex.irisa.fr/
    Data license: CC0 1.0 - Public domain, No copyright
    Data samples/events: 1k-500k
    Data dimensionality: 128

    Attributes:
        name: Human readable name for the dataset 
        path: Filepath for the root directory of dataset files
        trunc: Truncation of the data to specify the number of base and query vectors used
        timings: Number of batches of queries to collect runtimes over
        mode: Dataset mode specified by configuration files to enable/disable following attributes
        freq: Relative frequency of index queries and index updates
        lerp: Degree of interpolation between consecutive datapoints [0.0,1.0]

    Methods:
        __init__: Initialising internal parameters
        evaluate: Performance on a simulated dataset of that is continuously growing over time
        pregen: Download dataset files and computing groundtruth data using exhaustive searches
        vecs_train: Load training set of vectors used to tune ANN algorithms that require it
        vecs_base: Load base set of vectors used to initialise each ANN algorithm
        vecs_query: Load query set of vectors used to evaluate each ANN algorithm
        groundtruth: Load groundtruth indices used to evaluate each ANN algorithm
    """
    
    # Adapted from annbench https://github.com/matsui528/annbench/blob/main/annbench/dataset/sift1m.py
    
    def __init__(self, cfg):
        self.name = cfg.data.name
        self.path = Path(cfg.data.path)
        self.mode = cfg.data.mode
        self.trunc = cfg.data.scale
        self.freq = 1
        if self.mode == "efreq" or self.mode == "esfreq":
            self.freq = cfg.data.scale
            self.trunc = cfg.data.trunc
        self.timings = cfg.data.timings
        self.lerp = 0.0
        if self.mode == 'lerp':
            self.lerp = cfg.data.lerp
        
    def evaluate(self, algo, cfg):
        # Load queries
        vecs = self.vecs_query()
        # Initialise parameters
        nq = int(vecs.shape[0] / 2)
        ngt = 10000
        if self.trunc < 10:
            ngt = 100
        elif self.trunc < 100:
            ngt = 1000
        # Initialise results
        ids = -1 * np.ones([ngt, cfg.topk]).astype('int') # Run all queries, store gt indices only
        ts = np.zeros([self.timings, 3])
        idi, ti = 0, 0
        id = -1 * np.ones([cfg.topk])
        # Run benchmark
        for query in range(nq, nq*2, self.freq):
            # Get sample
            if self.lerp > 0:
                vecs[query] = lerp(vecs[query], vecs[query-1], self.lerp)
            # Process queries
            t0 = time.time()
            if self.mode == "es_freq":
                id = algo.query(vecs=np.array(vecs[query:query+self.freq,:]), topk=cfg.topk, cfg=cfg)
                ts[ti,0] = ts[ti,0] + time.time() - t0
                id = np.array(id[0])
            else:
                id = algo.query(vecs=np.array([vecs[query]]), topk=cfg.topk, cfg=cfg)
                ts[ti,0] = ts[ti,0] + time.time() - t0
            if (query) % (nq / ngt) <= (query - self.freq) % (nq / ngt):
                id = np.array(id)
                ids[idi,:len(id.squeeze())] = id
                idi = idi + 1
            # Process add events
            t0 = time.time()
            algo.add(vecs=vecs[:query+self.freq], start=query, count=self.freq)
            ts[ti,1] = ts[ti,1] + time.time() - t0
            if (query - nq + self.freq) % (nq / self.timings) <= (query - nq) % (nq / self.timings):
                ts[ti,:2] = ts[ti,:2] * self.timings / nq
                ts[ti,2] = algo.get_memory_usage(cfg.mem_type)
                ti = ti + 1
        # Return results
        return ts, ids

    def pregen(self, cfg):
        # Download data blobs
        if not self.path.exists():
            self.path.mkdir(parents=True)
            tar_path = self.path / "sift.tar.gz"
            urlretrieve("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", tar_path)
        if (self.path / "sift/sift_base.fvecs").exists() and \
           (self.path / "sift/sift_learn.fvecs").exists():
           pass
        else:
            with tarfile.open(tar_path, 'r:gz') as f:
                f.extractall(path=self.path)
        # Check for groundtruth files
        gt_path = self.path / f"sift/{self.name}_{self.mode}{self.trunc}_{self.freq}_gt.ivecs"
        if not gt_path.exists():
            # Initialise bruteforce algorithm
            from ..algo.linear import LinearANN
            algo = LinearANN()
            algo.init(D=self.D(), maxN=2000*self.trunc, cfg=cfg)
            vecs = self.vecs_base()
            algo.add(vecs=self.vecs_base(), start=0, count=vecs.shape[0])
            # Generate groundtruth
            _, ids = self.evaluate(algo=algo, cfg=cfg)
            ivecs_write(gt_path, ids)

    def vecs_train(self):
        vec_path = self.path / "sift/sift_learn.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def vecs_base(self):
        vec_path = self.path / "sift/sift_base.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))[:1000*self.trunc,:]

    def vecs_query(self):
        vec_path = self.path / "sift/sift_base.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))[:2000*self.trunc,:]

    def groundtruth(self):
        gt_path = self.path / f"sift/{self.name}_{self.mode}{self.trunc}_{self.freq}_gt.ivecs"
        assert gt_path.exists()
        return ivecs_read(fname=str(gt_path))
