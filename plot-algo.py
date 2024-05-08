from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
from dyann.util import stringify_dict
from dyann.vis import draw_loglog, draw_series

# Initialise message logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(name)s: %(message)s")
log = logging.getLogger(__name__)

def main():
    # Load base configuration values
    cfg_path = Path(".").joinpath("conf/plot.yaml")
    if not cfg_path.exists():
        log.info(f"No config at {cfg_path}")
        return
    default_cfg = OmegaConf.load(cfg_path)
    base_cfg = OmegaConf.merge(default_cfg, OmegaConf.from_cli())
    log.info(OmegaConf.to_yaml(base_cfg))

    timestamp = f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
    out = Path(base_cfg.output)
    img = Path(base_cfg.img_out)
    img.mkdir(exist_ok=True, parents=True)  # Make sure the img directory exists

    for p_dataset in sorted(out.glob("*")):
        if p_dataset.is_file() or not p_dataset.name in base_cfg.data:
            continue
        runtime_vs_recall, recall_series, recall_series, runtime_series, buildtime_series, searchtime_series = [], [], [], [], [], []
        for p_algo in sorted(p_dataset.glob("*")):
            if p_algo.is_file() or not p_algo.name in base_cfg.algo:
                continue
            for p_result in sorted(p_algo.glob("*")):
                if not p_result.is_file():
                    continue
                log.info(f"Reading {p_result}")
                with p_result.open("rt") as f:
                    ret_all = yaml.safe_load(f)
                for ret in ret_all:
                    # "ret" is for one param_build. "ret" contains several results for each param_query
                    recall, runtime, buildtime, searchtime, ctrls = [], [], [], [], []
                    for r in ret:
                        recall.append(r["recall"][0]) #top50
                        runtime.append(r["runtime_per_query"])
                        buildtime.append(r["buildtime_per_query"])
                        searchtime.append(r["searchtime_per_query"])
                        ctrls.append(list(r['param_query'].values())[0])  # Just extract a value
                    line = {
                        "xs": np.array(recall), "ys": 1.0 / np.array(runtime), "ctrls": ctrls,
                        "ctrl_label": list(ret[0]['param_query'])[0],  # Just extract the name of query param
                        "label": p_algo.name + "(" + stringify_dict(d=ret[0]['param_build']) + ")"
                    }
                    runtime_vs_recall.append(line.copy())
                    moving_avg = line.pop("xs")
                    window = int(base_cfg.window * len(moving_avg[0,:]))
                    for i in range(len(moving_avg)):
                        cumsum = np.cumsum(np.insert(moving_avg[i,:], 0, np.repeat(moving_avg[i,0], window)))
                        moving_avg[i,:] = (cumsum[window:] - cumsum[:-window]) / float(window)
                    line["ys"] = moving_avg
                    recall_series.append(line.copy())
                    moving_avg = np.array(runtime)
                    window = int(base_cfg.window * len(moving_avg[0,:]))
                    for i in range(len(moving_avg)):
                        cumsum = np.cumsum(np.insert(moving_avg[i,:], 0, np.repeat(moving_avg[i,0], window)))
                        moving_avg[i,:] = (cumsum[window:] - cumsum[:-window]) / float(window)
                    line["ys"] = moving_avg
                    runtime_series.append(line.copy())
                    moving_avg = np.array(buildtime)
                    window = int(base_cfg.window * len(moving_avg[0,:]))
                    for i in range(len(moving_avg)):
                        cumsum = np.cumsum(np.insert(moving_avg[i,:], 0, np.repeat(moving_avg[i,0], window)))
                        moving_avg[i,:] = (cumsum[window:] - cumsum[:-window]) / float(window)
                    line["ys"] = moving_avg
                    buildtime_series.append(line.copy())
                    moving_avg = np.array(searchtime)
                    window = int(base_cfg.window * len(moving_avg[0,:]))
                    for i in range(len(moving_avg)):
                        cumsum = np.cumsum(np.insert(moving_avg[i,:], 0, np.repeat(moving_avg[i,0], window)))
                        moving_avg[i,:] = (cumsum[window:] - cumsum[:-window]) / float(window)
                    line["ys"] = moving_avg
                    searchtime_series.append(line.copy())

        log.info(f"Writing to {img.resolve()}")

        # Save the images to the result_img directory
        draw_loglog(lines=runtime_vs_recall, xlabel="recall", ylabel="query/sec", title=f"Parameter sweep {p_dataset.name}",
            filename=img / f"sweep-params-{p_dataset.name}-{timestamp}.png", with_ctrl=base_cfg.with_query_param, with_error=False, width=base_cfg.width, height=base_cfg.height)
        draw_series(lines=recall_series, ylabel=f"recall (moving avg)", title=f"Recall series {p_dataset.name}",
            filename=img / f"sweep-recall-{p_dataset.name}-{timestamp}.png", with_ctrl=base_cfg.with_query_param, width=base_cfg.width, height=base_cfg.height)
        draw_series(lines=runtime_series, ylabel="runtime (moving avg)", title=f"Runtime series {p_dataset.name}",
            filename=img / f"sweep-runtime-{p_dataset.name}-{timestamp}.png", with_ctrl=base_cfg.with_query_param, width=base_cfg.width, height=base_cfg.height)
        draw_series(lines=buildtime_series, ylabel="build time (moving avg)", title=f"Buildtime series {p_dataset.name}",
            filename=img / f"sweep-buildtime-{p_dataset.name}-{timestamp}.png", with_ctrl=base_cfg.with_query_param, width=base_cfg.width, height=base_cfg.height)
        draw_series(lines=searchtime_series, ylabel="search time (moving avg)", title=f"Searchtime series {p_dataset.name}",
            filename=img / f"sweep-searchtime-{p_dataset.name}-{timestamp}.png", with_ctrl=base_cfg.with_query_param, width=base_cfg.width, height=base_cfg.height)

        log.info("Finished writing")

if __name__ == "__main__":
    main()
