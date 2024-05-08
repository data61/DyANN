def instantiate_dataset(cfg):
    """ Imports and instantiate a dataset class

    Parameters:
        cfg: configuration object containing the name of the target dataset
                selected from {datacol, featlearn}
    Returns:
        an instance of the specified dataset class or None object if name is invalid
    """
    if cfg.data.name == "datacol":
        from .datacol import OnlineDataCollection
        return OnlineDataCollection(cfg=cfg)
    elif cfg.data.name == "featlearn":
        from .featlearn import OnlineFeatureLearning
        return OnlineFeatureLearning(cfg=cfg)
    return None



