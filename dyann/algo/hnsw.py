from .base import BaseANN
import numpy as np
import hnswlib

# Refer to https://github.com/nmslib/hnswlib/blob/master/README.md
# Adapted from https://github.com/matsui528/annbench/blob/main/annbench/algo/hnsw.py

class HnswANN(BaseANN):
    def __init__(self):
        super().__init__()
        self.ef_construction, self.M, self.index = None, None, None

    def init(self, D, maxN, cfg):
        super().init(D=D, maxN=maxN, cfg=cfg)
        self.ef_construction = cfg.algo.build.ef_construction
        self.M =cfg.algo.build.M
        self.maxN = maxN
        self.index = hnswlib.Index(space='l2', dim=D)
        self.index.set_num_threads(1)
        self.index.init_index(max_elements=self.maxN, ef_construction=self.ef_construction, M=self.M)

    def has_train(self):
        return False

    def do_add(self, vecs, start, count):
        self.index.add_items(data=vecs[start:start+count,:], ids=np.array(range(start, start+count)))

    def query(self, vecs, topk, cfg):
        self.index.set_num_threads(1)
        self.index.set_ef(ef=cfg.algo.query.ef)
        try:
            labels, _ = self.index.knn_query(data=vecs, k=topk)
        except RuntimeError:
            labels = -1 * np.ones([vecs.shape[0], topk])
        return labels


