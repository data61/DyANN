from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from itertools import cycle
from dyann.util import stringify_dict

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
    csv = Path(base_cfg.csv_out)
    csv.mkdir(exist_ok=True, parents=True)  # Make sure the csv directory exists
    topk = 50
    topi = 0

    for dataset in base_cfg.data:
        max_pareto = []
        for algo in base_cfg.algo:
            line = { "xs": [], "ys": [], "ctrls": [], "label": algo }
            best = 0
            line_all = []
            for result in sorted(out.glob(f"{dataset}/{algo}/*")):
                if not result.is_file():
                    continue
                log.info(f"Reading {result}")
                with result.open("rt") as f:
                    ret_all = yaml.safe_load(f)
                for ret in ret_all: # "ret" is for one param_build. "ret" contains several results for each param_query
                    for r in ret:
                        recall = np.mean(np.array(r["recall"][topi]) / topk)
                        runtime = np.mean(1.0 / np.array(r["runtime_per_query"]))
                        if len(max_pareto) > 0:
                            runtime = runtime / max_pareto[0]["ys"][0]
                            ctrls = stringify_dict(d=ret[0]['param_build']) + ',' + stringify_dict(r['param_query'])
                            line_all.append([recall, runtime, ctrls])
                        else:
                            harmonic_mean = 2 * recall * runtime / (recall + runtime)
                            if harmonic_mean > best:
                                line["xs"] = np.array([recall])
                                line["ys"] = np.array([runtime])
                                line["ctrls"] = [""]
                                best = harmonic_mean
            if len(max_pareto) > 0:
                line_trim = [[],[],[]]
                while (len(line_all) > 0):
                    min_recall = 0
                    for i,r in enumerate(line_all[1:]):
                        if r[0] < line_all[min_recall][0]:
                            min_recall = i + 1
                    min_recall = line_all.pop(min_recall)
                    is_pareto = True
                    for r in line_all:
                        if r[1] > min_recall[1]:
                            is_pareto = False
                    if is_pareto:
                        line_trim[0].append(min_recall[0])
                        line_trim[1].append(min_recall[1])
                        line_trim[2].append(min_recall[2])
                line["xs"] = np.array(line_trim[0])
                line["ys"] = np.array(line_trim[1])
                line["ctrls"] = line_trim[2]
            if len(line["xs"]) > 0:
                max_pareto.append(line)

        if len(max_pareto) < 2:
            continue


        # Save the images to the result_img directory
        log.info(f"Writing {dataset} plot to {img.resolve()}")

        colors = 20
        angle = np.array(range(colors))*2*np.pi/colors
        pallet = np.array([0.4+0.4*abs(np.sin(angle-np.pi/3))-0.3*np.cos(angle-np.pi/3),
                            0.55+0.2*np.cos(2*angle+np.pi/3)-0.2*np.cos(angle+np.pi/3),
                            0.55+0.2*np.cos(2*angle+np.pi/3)-0.2*np.cos(angle+3*np.pi/4)]).T
        color = cycle(pallet[::int(colors/(len(max_pareto)-1))])
        marker = cycle(['o','s','^','v','p','d','<','>'])
        plt.figure(figsize=(base_cfg.width, base_cfg.height))
        for line in max_pareto[1:]:
            plt.plot(line["xs"], line["ys"], label=line["label"], color=next(color), marker=next(marker), linestyle="-")
            for i, [x, y, ctrl] in enumerate(zip(line["xs"], line["ys"], line["ctrls"])):
                plt.annotate(text=f"{ctrl}", xy=(x, y), xytext=(0, 5), textcoords="offset pixels")

        plt.xlabel(f"Recall@{topk}")
        plt.ylabel(f"Speedup over {max_pareto[0]['label']}")
        plt.grid(which="both")
        #plt.yscale("log")
        plt.axis("tight")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.title(f"Pareto front {dataset}")
        plt.savefig(img / f"pareto-{dataset}-{timestamp}.png", bbox_inches="tight")
        plt.cla()

        # Write csv file
        log.info(f"Writing {dataset} csv to {csv.resolve()}")

        text = [""]
        for i,line in enumerate(max_pareto[1:]):
            if i > 0:
                text[0] += ","
            text[0] += f"{line['label']}-x,{line['label']}-y"
            for j,point in enumerate(line["xs"]):
                if (j >= len(text) - 1):
                    text.append("".join([("," if x == 0 else ",,") for x in range(i)]))
                if i > 0:
                    text[j+1] += ","
                text[j+1] += f"{point},{line['ys'][j]}"

        with open(csv / f"pareto-{dataset}-{timestamp}.csv", "w") as f:
            for row in text:
                f.write(f"{row}\r\n")

    log.info("Finished writing")

if __name__ == "__main__":
    main()
