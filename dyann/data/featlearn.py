from .base import BaseDataset
from pathlib import Path
import subprocess
import numpy as np
import time
from ..util import ivecs_read, fvecs_read, ivecs_write, lerp

class OnlineFeatureLearning(BaseDataset):
    """
    A class for simulation of online feature learning datasets

    Data source: https://github.com/matsui528/deep1b_gt#bonus-deep1m
    Data license: MIT - Modification, Distribution, Private use, Commercial use
    Data samples/events: 1k-500k
    Data dimensionality: 96

    Attributes:
        name: Human readable name for the dataset 
        path: Filepath for the root directory of dataset files
        trunc: Truncation of the data to specify the number of base and query vectors used
        epoch: Number of epochs to simulate and collect runtimes over
        batch: Number of batches of index queries and index updates within each epoch
        mode: Dataset mode specified by configuration files to enable/disable following attributes
        freq: Relative frequency of index queries and index updates
        lerp: Degree of interpolation between consecutive datapoints [0.0,1.0]

    Methods:
        __init__: Initialising internal parameters
        evaluate: Performance on a simulated dataset of updates to a feature embedding space
        pregen: Download dataset files and computing groundtruth data using exhaustive searches
        vecs_train: Load training set of vectors used to tune ANN algorithms that require it
        vecs_base: Load base set of vectors used to initialise each ANN algorithm
        vecs_query: Load query set of vectors used to evaluate each ANN algorithm
        groundtruth: Load groundtruth indices used to evaluate each ANN algorithm
    """
    
    # Adapted from annbench https://github.com/matsui528/annbench/blob/main/annbench/dataset/deep1m.py
    
    def __init__(self, cfg):
        self.name = cfg.data.name
        self.path = Path(cfg.data.path)
        self.mode = cfg.data.mode
        self.trunc = cfg.data.scale
        self.freq = cfg.data.batch
        if self.mode == "efreq" or self.mode == "esfreq":
            self.freq = cfg.data.scale
            self.trunc = cfg.data.trunc
        self.epochs = cfg.data.epochs
        self.batch = cfg.data.batch
        if self.mode == "esfreq":
            self.batch = cfg.data.scale
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
        ts = np.zeros([self.epochs, 3])
        idi = 0
        # Run benchmark
        for epoch in range(self.epochs):
            for b,batch in enumerate(range(0, nq, self.batch)):
                # Update samples
                target = nq
                if self.mode == 'lerp':
                    target = target + batch
                update = lerp(vecs[batch:batch+self.batch], vecs[target:target+self.batch], self.lerp)
                # Process queries
                t0 = time.time()
                id = algo.query(vecs=np.array(update[:self.batch,:]), topk=cfg.topk, cfg=cfg)
                ts[epoch,0] = ts[epoch,0] + time.time() - t0
                if b < ngt / self.epochs:
                    id = np.array(id[0])
                    ids[idi,:len(id.squeeze())] = id
                    idi = idi + 1
                vecs[batch:batch+self.batch] = update
                # Process update events
                t0 = time.time()
                for f in range(batch, batch+self.batch, self.freq):
                    algo.update(vecs=vecs[:nq], start=f, count=self.freq)
                ts[epoch,1] = ts[epoch,1] + time.time() - t0
            ts[epoch,:2] = ts[epoch,:2] / nq
            ts[epoch,2] = algo.get_memory_usage(cfg.mem_type)
        # Return results
        return ts, ids

    def pregen(self, cfg):
        # Download data blobs
        root = str(self.path.resolve())
        if not self.path.exists():
            self.path.mkdir(parents=True)
            subprocess.run(f"git clone https://github.com/matsui528/deep1b_gt.git {root}", shell=True) # https://github.com/matsui528/deep1b_gt#bonus-deep1m
        if (self.path / "deep1b/deep1M_base.fvecs").exists() and \
           (self.path / "deep1b/deep1M_learn.fvecs").exists():
           pass
        else:
            # Download base_00, learn_00, and query on {root}/deep1b. This may take some hours. I recommend preparing 25GB of the disk space.
            subprocess.run(f"python {root}/download_deep1b.py --root {root}/deep1b --base_n 1 --learn_n 1 --ops query base learn", shell=True)
            # Select top 1M vectors from base_00 and save it on deep1M_base.fvecs
            subprocess.run(f"python {root}/pickup_vecs.py --src {root}/deep1b/base/base_00 --dst {root}/deep1b/deep1M_base.fvecs --topk 1000000", shell=True)
            # Select top 100K vectors from learn_00 and save it on deep1M_learn.fvecs
            subprocess.run(f"python {root}/pickup_vecs.py --src {root}/deep1b/learn/learn_00 --dst {root}/deep1b/deep1M_learn.fvecs --topk 100000", shell=True)
        # Check for groundtruth files
        gt_path = self.path / f"deep1b/{self.name}_{self.mode}{self.trunc}_{self.freq}_gt.ivecs"
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
        vec_path = self.path / "deep1b/deep1M_learn.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def vecs_base(self):
        vec_path = self.path / "deep1b/deep1M_base.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))[:1000*self.trunc,:]

    def vecs_query(self):
        vec_path = self.path / "deep1b/deep1M_base.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))[:2000*self.trunc,:]

    def groundtruth(self):
        gt_path = self.path / f"deep1b/{self.name}_{self.mode}{self.trunc}_{self.freq}_gt.ivecs"
        assert gt_path.exists()
        return ivecs_read(fname=str(gt_path))
