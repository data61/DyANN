from .base import BaseANN
import numpy as np
import faiss

# Refer to https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVF.h
# Adapted from https://github.com/matsui528/annbench/blob/main/annbench/algo/faiss_cpu.py

class IvfpqANN(BaseANN):
    def __init__(self):
        super().__init__()
        self.M, self.nlist, self.index = None, None, None

    def init(self, D, maxN, cfg):
        super().init(D=D, maxN=maxN, cfg=cfg)
        self.M, self.nlist = cfg.algo.build.M, cfg.algo.build.nlist
        faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        quantizer = faiss.IndexFlatL2(D)
        self.index = faiss.IndexIVFPQ(quantizer, D, self.nlist, self.M, 8)

    def has_train(self):
        return True

    def train(self, vecs):
        self.index.train(vecs)

    def do_add(self, vecs, start, count):
        self.index.add_with_ids(vecs[start:start+count,:], np.array(range(start, start+count)))

    def query(self, vecs, topk, cfg):
        self.index.nprobe = cfg.algo.query.nprobe
        _, ids = self.index.search(x=vecs, k=topk)
        return ids

class Ivfpq4bitANN(IvfpqANN):
    def init(self, D, maxN, cfg):
        super().init(D=D, maxN=maxN, cfg=cfg)
        faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        quantizer = faiss.IndexFlatL2(D)
        self.index = faiss.IndexIVFPQ(quantizer, D, self.nlist, self.M, 4)
