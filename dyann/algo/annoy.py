from .base import BaseANN
import annoy

# Refer to https://github.com/spotify/annoy/blob/master/src/annoylib.h
# Adapted from https://github.com/matsui528/annbench/blob/main/annbench/algo/annoy.py

class AnnoyANN(BaseANN):
    def __init__(self):
        super().__init__()
        self.n_trees, self.index = None, None

    def init(self, D, maxN, cfg):
        super().init(D=D, maxN=maxN, cfg=cfg)
        self.n_trees = cfg.algo.build.n_trees
        self.index = annoy.AnnoyIndex(f=D, metric="euclidean")

    def has_train(self):
        return False

    def do_add(self, vecs, start, count):
        for n, vec in enumerate(vecs[start:start+count]):
            self.index.add_item(n + start, vec.tolist())
        self.index.unbuild()
        self.index.build(self.n_trees, n_jobs=1)

    def query(self, vecs, topk, cfg):
        return [self.index.get_nns_by_vector(vector=vec.tolist(), n=topk, search_k=cfg.algo.query.search_k) for vec in vecs]

