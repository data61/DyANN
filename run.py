# File and data handling 
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import yaml
import numpy as np
# Memory and runtime monitoring
from datetime import datetime
import time
import gc
import tracemalloc
# Internal functions
from dyann.algo.proxy import instantiate_algorithm
from dyann.data.proxy import instantiate_dataset
from dyann.util import recall_at_r

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
    log.info(OmegaConf.to_yaml(base_cfg))

    # Sweep algorithms
    for algo_name in base_cfg.algo:
        # Instantiate a search algorithm class
        cfg_path = Path(".").joinpath(f"conf/algo/{algo_name}_build.yaml")
        if not cfg_path.exists():
            log.info(f"Skipping algorithm {algo_name} - no config at {cfg_path}")
            continue
        algo_cfg = OmegaConf.create(base_cfg)
        algo_cfg.algo = {}
        algo_cfg = OmegaConf.merge(algo_cfg, OmegaConf.load(cfg_path))
        algo = instantiate_algorithm(cfg=algo_cfg)
        cfg_path = Path(".").joinpath(f"conf/algo/{algo_name}_search.yaml")
        algo_cfg = OmegaConf.merge(algo_cfg, OmegaConf.load(cfg_path))
        if not cfg_path.exists():
            log.info(f"Skipping algorithm {algo_name} - no config at {cfg_path}")
            continue
        # Sweep datasets
        for data_name in base_cfg.data:
            # Instantiate a dataset class
            cfg_path = Path(".").joinpath(f"conf/data/{data_name}.yaml")
            if not cfg_path.exists():
                log.info(f"Skipping dataset {data_name} - no config at {cfg_path}")
                continue
            data_cfg = OmegaConf.create(algo_cfg)
            data_cfg.data = {}
            data_cfg = OmegaConf.merge(data_cfg, OmegaConf.load(cfg_path))
            # Sweep dataset scale parameters
            for scale in data_cfg.data.scale:
                scale_cfg = OmegaConf.create(data_cfg)
                scale_cfg.data.scale = scale
                # Load base dataset
                dataset = instantiate_dataset(cfg=scale_cfg)
                # Pregenerate dataset values
                pregen_cfg = OmegaConf.create(scale_cfg)
                pregen_cfg.algo = {}
                pregen_path = Path(".").joinpath("./conf/algo/linear_build.yaml")
                pregen_cfg = OmegaConf.merge(pregen_cfg, OmegaConf.load(pregen_path))
                pregen_cfg.algo.build = pregen_cfg.algo.build[0]
                dataset.pregen(cfg=pregen_cfg)
                base_vecs = dataset.vecs_base()
                base_size = base_vecs.shape[0]
                # Sweep algorithm build and update parameters
                ret_all = []
                for build in data_cfg.algo.build:
                    build_cfg = OmegaConf.create(scale_cfg)
                    build_cfg.algo.build = build
                    # Sweep algorithm search parameters
                    ret = []
                    for query in build_cfg.algo.query:
                        query_cfg = OmegaConf.create(build_cfg)
                        query_cfg.algo.query = query

                        # Build the index
                        log.info(f"Start to build with {build}")
                        if base_cfg.mem_type == "trc_mem" or base_cfg.mem_type == "trc_peak":
                            gc.collect()
                            tracemalloc.start()
                        m0 = algo.get_memory_usage(base_cfg.mem_type)
                        t0 = time.time()
                        algo.init(D = base_vecs.shape[1], maxN = base_size * 2, cfg = build_cfg)
                        if algo.has_train():
                            log.info("Start to train")
                            algo.train(vecs=dataset.vecs_train())
                        log.info("Start to add")
                        algo.do_add(vecs=base_vecs, start = 0, count = base_size)

                        t1 = time.time()
                        m1 = algo.get_memory_usage(base_cfg.mem_type)
                        buildtime_per_base = (t1 - t0) / base_size
                        memory_per_base = (m1 - m0) / base_size

                        if base_cfg.mem_type == "trc_mem" or base_cfg.mem_type == "trc_peak":
                            tracemalloc.stop()
                        
                        # Search the index
                        log.info(f"Start to search with {query}")
                        runtime, ids = dataset.evaluate(algo, query_cfg)
                        recall = [recall_at_r(I=ids, gt=dataset.groundtruth(), r=r) for r in range(base_cfg.topk,0,-20)]
                        searchtime_per_query = runtime[:,0]
                        buildtime_per_query = runtime[:,1]
                        runtime_per_query = [x+y for x,y in zip(searchtime_per_query, buildtime_per_query)]
                        memory_query = runtime[:,2]

                        # Compile results
                        ret.append({
                            "param_build": dict(build),
                            "buildtime_per_base": float(buildtime_per_base),
                            "memory_per_base": float(memory_per_base),
                            "param_query": dict(query),
                            "runtime_per_query": [float(x) for x in runtime_per_query],
                            "searchtime_per_query": [float(x) for x in searchtime_per_query],
                            "buildtime_per_query": [float(x) for x in buildtime_per_query],
                            "memory_query": [float(x) for x in memory_query],
                            "recall": [[float(x) for x in y] for y in recall]
                        })
                        log.info("Finish")

                    ret_all.append(ret)

                # Save results to output directory
                out_path = Path(f"{base_cfg.output}/{data_name}/{algo_name}/result-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.yaml")
                out_path.parent.mkdir(exist_ok=True, parents=True)
                with out_path.open("wt") as f:
                    yaml.dump(ret_all, f)

if __name__ == "__main__":
    main()
