from .base import BaseANN
import scann

# Some hypter-parameters are from https://github.com/facebookresearch/faiss/blob/master/benchs/bench_all_ivf/cmp_with_scann.py
# This SCANN does not include re-order process

# Refer to https://github.com/google-research/google-research/blob/master/scann/scann/scann_ops/py/scann_builder.py
# Adapted from https://github.com/matsui528/annbench/blob/main/annbench/algo/scann.py

class ScannANN(BaseANN):
    def __init__(self):
        super().__init__()
        self.num_leaves, self.reorder, self.index = None, None, None

    def init(self, D, maxN, cfg):
        super().init(D=D, maxN=maxN, cfg=cfg)
        self.num_leaves = cfg.algo.build.num_leaves # ~ sqrt(N)
        self.reorder = cfg.algo.build.reorder

    def has_train(self):
        return False

    def do_add(self, vecs, start, count):
        sb = scann.scann_ops_pybind.builder(db=vecs[:start+count], num_neighbors=10, distance_measure="squared_l2")
        sb.set_n_training_threads(1)
        sb.tree(num_leaves=self.num_leaves, num_leaves_to_search=100, training_sample_size=min(start+count, 250000))
        sb.score_ah(dimensions_per_block=2, anisotropic_quantization_threshold=0)

        # Re-compute based on the actual vectors
        if self.reorder:
            sb.reorder(self.reorder)

        self.index = sb.build()

    def do_update(self, vecs, start, count):
        self.do_add(vecs, 0, vecs.shape[0])


    def query(self, vecs, topk, cfg):
        ids, _ = self.index.search_batched(vecs, leaves_to_search=cfg.algo.query.nprobe, final_num_neighbors=topk)
        # Note: There exists a function .search_batched_parallel() as well.
        return ids

