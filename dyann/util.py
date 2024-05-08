import numpy as np

def lerp(vecs, target, frac):
    """Linerly interpolates between two vectors"""
    assert vecs.shape[0] == target.shape[0]
    if len(vecs.shape) == 2:
        assert vecs.shape[1] == target.shape[1]
    else:
        assert len(vecs.shape) == len(target.shape)
    return (1 - frac) * vecs + frac * target

def recall_at_r(I, gt, r):
    """ Compute Recall@r over a set of neighbour indices

    Parameters:
        I: Retrieval result indices
        gt: Groundtruth indices
        r (int): Top-r
    Returns:
        The recall@r over all indies
    """
    assert r <= I.shape[1]
    assert r <= gt.shape[1]
    assert I.shape[1] >= gt.shape[1]
    assert len(I) == len(gt)
    n_ok = []
    for i in range(I.shape[0]):
        n_ok.append(len(list(set(I[i, :]) & set(gt[i, :r]))))
    return n_ok

# The following fuinctions are from annbench
# https://github.com/matsui528/annbench/blob/main/annbench/util.py

def stringify_dict(d):
    """ d = {"a", 123, "b", "xyz", "c": "hij"} -> "a=1, b=xyz, c=hij" """
    if len(d) == 0:
        return ""
    assert isinstance(d, dict)
    s = ""
    for k, v in d.items():
        s += str(k) + "=" + str(v) + ", "
    return s[:-2]  # delete the last ", "

# The following functions are from faiss
# https://github.com/facebookresearch/faiss/blob/master/benchs/datasets.py

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)