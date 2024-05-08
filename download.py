from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from dyann.data.proxy import instantiate_dataset

# Initialise message logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s: %(message)s")
log = logging.getLogger(__name__)

def main():
    # Load base configuration values
    cfg_path = Path(".").joinpath("conf/run.yaml")
    if not cfg_path.exists():
        log.info(f"No config at {cfg_path}")
        return
    default_cfg = OmegaConf.load(cfg_path)
    base_cfg = OmegaConf.merge(default_cfg, OmegaConf.from_cli())
    base_cfg.algo = "linear"
    cfg_path = Path("./").joinpath(f"conf/algo/{base_cfg.algo}_build.yaml")
    if not cfg_path.exists():
        log.info(f"No config at {cfg_path}")
        return
    base_cfg.algo = {}
    base_cfg = OmegaConf.merge(base_cfg, OmegaConf.load(cfg_path))
    base_cfg.algo.build = base_cfg.algo.build[0]
    log.info(OmegaConf.to_yaml(base_cfg))
    
    # Sweep datasets
    for data_name in base_cfg.data:
        # Load dataset configuration values
        cfg_path = Path(".").joinpath(f"conf/data/{data_name}.yaml")
        if not cfg_path.exists():
            log.info(f"Skipping dataset {data_name} - no config at {cfg_path}")
            continue
        data_cfg = OmegaConf.create(base_cfg)
        data_cfg.data = {}
        data_cfg = OmegaConf.merge(data_cfg, OmegaConf.load(cfg_path))
        # Sweep dataset scale parameters
        for scale in data_cfg.data.scale:
            scale_cfg = OmegaConf.create(data_cfg)
            scale_cfg.data.scale = scale
            dataset = instantiate_dataset(cfg=scale_cfg)
            log.info(f"Pregenerating {scale_cfg.data.name}_{scale} at {scale_cfg.data.path}")
            dataset.pregen(cfg=scale_cfg)
    log.info("Done")

if __name__ == "__main__":
    main()
