import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import cycle
import numpy as np

colors = 100
angle = np.array(range(colors))*2*np.pi/colors
pallet = np.array([0.4+0.4*abs(np.sin(angle-np.pi/3))-0.3*np.cos(angle-np.pi/3),
                    0.55+0.2*np.cos(2*angle+np.pi/3)-0.2*np.cos(angle+np.pi/3),
                    0.55+0.2*np.cos(2*angle+np.pi/3)-0.2*np.cos(angle+3*np.pi/4)]).T

def draw_loglog(lines, xlabel, ylabel, title, filename, with_ctrl, with_error, width, height):
    """
    Visualize search results and save them as an image
    Args:
        lines (list): search results. list of dict.
        xlabel (str): label of x-axis, usually "recall"
        ylabel (str): label of y-axis, usually "query per sec"
        title (str): title of the result_img
        filename (str): output file name of image
        with_ctrl (bool): show control parameters or not
        width (int): width of the figure
        height (int): height of the figure

    """
    color = cycle(pallet[::max(int(colors/len(lines)),1)])
    marker = cycle(['o','s','^','v','p','d','<','>'])

    plt.figure(figsize=(width, height))
    for line in lines:
        for key in ["xs", "ys", "label", "ctrls", "ctrl_label"]:
            assert key in line

    for line in lines:
        xs = np.array(line["xs"])
        ys = np.array(line["ys"])
        if len(xs.shape) == 2 and len(ys.shape) == 2:
            x_avg = np.mean(xs, axis=1)
            y_avg = np.mean(ys, axis=1)
            x_range = np.append([x_avg - np.min(xs,axis=1)], [np.max(xs, axis=1) - x_avg], axis=0)
            y_range = np.append([y_avg - np.min(ys,axis=1)], [np.max(ys, axis=1) - x_avg], axis=0)
            xs = x_avg
            ys = y_avg
            if with_error:
                plt.errorbar(xs, ys, yerr=y_range, xerr=x_range, label=line["label"], color=next(color), marker=next(marker), linestyle="None")
            else:
                plt.plot(xs, ys, label=line["label"], color=next(color), marker=next(marker), linestyle="-")
        else:
            plt.plot(xs, ys, label=line["label"], color=next(color), marker=next(marker), linestyle="-")
        if with_ctrl:
            for i, [x, y, ctrl] in enumerate(zip(xs, ys, line["ctrls"])):
                if (i == 0) == (i == len(xs)-1):
                    continue
                plt.annotate(text=line["ctrl_label"] + ":" + str(ctrl), xy=(x, y), xytext=(0, 5), textcoords="offset pixels")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which="both")
    plt.axis('tight')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.cla()

def draw_series(lines, ylabel, title, filename, with_ctrl, width, height):
    color = cycle(pallet[::max(int(colors/len(lines)),1)])
    marker = cycle(['o','s','^','v'])
    
    plt.figure(figsize=(width, height))
    for line in lines:
        for key in ["ys", "label", "ctrls", "ctrl_label"]:
            assert key in line

    legend = []
    for i,line in enumerate(lines):
        ys = np.array(line["ys"])
        if not len(ys.shape) == 2:
            continue
        c = next(color)
        m = next(marker)
        legend.append(mlines.Line2D([], [], color=c, marker=m, label=line["label"]))
        for j, [y, ctrl] in enumerate(zip(ys, line["ctrls"])):
            plt.plot(list(range(len(y))), y, color=c, marker=m, linestyle="-", markevery=(int(i*len(y)/(len(lines)*10)),int(len(y)/10)))
            if with_ctrl:
                if (j == 0) == (j == len(ys)-1):
                    continue
                plt.annotate(text=line["ctrl_label"] + ":" + str(ctrl), xy=(1, y[0]), xytext=(0, 5), textcoords="offset pixels")

    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(which="both")
    #plt.yscale("log")
    plt.axis('tight')
    plt.legend(handles=legend, bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.cla()