from .base import BaseANN
import faiss

# Refer to https://github.com/facebookresearch/faiss/blob/main/faiss/IndexFlat.h
# Adapted from https://github.com/matsui528/annbench/blob/main/annbench/algo/faiss_cpu.py

class LinearANN(BaseANN):
    def __init__(self):
        super().__init__()
        self.index = None

    def init(self, D, maxN, cfg):
        super().init(D=D, maxN=maxN, cfg=cfg)
        faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        self.index = faiss.IndexFlatL2(D)

    def has_train(self):
        return False

    def do_add(self, vecs, start, count):
        self.index.add(vecs[start:start+count])

    def do_update(self, vecs, start, count):
        self.index.reset()
        self.index.add(vecs)

    def query(self, vecs, topk, cfg):
        _, ids = self.index.search(x=vecs, k=topk)
        return ids



