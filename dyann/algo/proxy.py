from omegaconf import OmegaConf

def instantiate_algorithm(cfg):
    """ Imports and instantiate an algorithm class

    Parameters:
        cfg: configuration object containing the name of the target algorithm
                selected from {linear, annoy, ivfpq, hnsw, scann, kdtree}
    Returns:
        an instance of the specified algorithm class or None object if name is invalid
    """

    if cfg.algo.name == "linear":
        from .linear import LinearANN
        return LinearANN()
    elif cfg.algo.name == "annoy":
        from .annoy import AnnoyANN
        return AnnoyANN()
    elif cfg.algo.name == "ivfpq":
        from .ivfpq import Ivfpq4bitANN
        return Ivfpq4bitANN()
    elif cfg.algo.name == "hnsw":
        from .hnsw import HnswANN
        return HnswANN()
    elif cfg.algo.name == "scann":
        from .scann import ScannANN
        return ScannANN()
    elif cfg.algo.name == "kdtree":
        from .kdtree import KDTreeANN
        return KDTreeANN()
    else:
        return None



