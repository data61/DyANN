from .base import BaseANN
from sklearn.neighbors import KDTree

# Refer to https://scikit-learn.org/stable/modules/neighbors.html
class KDTreeANN(BaseANN):
    def __init__(self):
        super().__init__()
        self.num_leaves = None
        self.dual_tree = None
        self.bfs = None

    def init(self, D, maxN, cfg):
        super().init(D=D, maxN=maxN, cfg=cfg)
        self.num_leaves = cfg.algo.build.num_leaves
        self.dual_tree = True
        self.bfs = True
        self.maxN = maxN
        self.index = []

    def has_train(self):
        return False

    def do_add(self, vecs, start, count):
        self.index = KDTree(vecs[:start+count], leaf_size=self.num_leaves)

    def do_update(self, vecs, start, count):
        self.do_add(vecs, 0, self.maxN)

    def query(self, vecs, topk, cfg):
        return self.index.query(vecs, k=topk, return_distance=False, dualtree=self.dual_tree, breadth_first=self.bfs)

