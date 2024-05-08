import os
import psutil
import gc
import tracemalloc
import resource
import subprocess

class BaseANN(object):
    """ Base class for all ANN algorithms

    Defined Methods:
        get_memory_usage: helper function for memory footprint monitoring
        add: manage build latency when adding samples
        update: manage build latency when updating samples
    Inherited Methods:
        __init__: (optional) initialise internal parameters
        init: (optional) initialise the algorithm for a particular dataset
        has_train: does this algorithm need a training set
        train: (optional) train algorithm parameters on a training set
        do_add: add samples to the algorithms index
        do_update: (optional) update samples in the algorithms index
        query: search for ANNs using the algorithms index
    """

    def get_memory_usage(self, type):
        """Return the current memory usage of this algorithm instance"""
        gc.collect()
        if type == "psu_rss":
            return psutil.Process(os.getpid()).memory_info().rss
        if type == "psu_vms":
            return psutil.Process(os.getpid()).memory_info().vms
        if type == "psu_shr":
            return psutil.Process(os.getpid()).memory_info().shared
        if type == "res_rss":
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if type == "ps_rss":
            return int(subprocess.Popen(['ps','-o','rss','-p',str(os.getpid())],stdout=subprocess.PIPE).communicate()[0].decode().split(os.linesep)[1]) * 1024
        if type == "ps_vms":
            return int(subprocess.Popen(['ps','-o','vsize','-p',str(os.getpid())],stdout=subprocess.PIPE).communicate()[0].decode().split(os.linesep)[1]) * 1024
        if type == "ps_mem":
            return int(subprocess.Popen(['ps','-o','size','-p',str(os.getpid())],stdout=subprocess.PIPE).communicate()[0].decode().split(os.linesep)[1]) * 1024
        if type == "slurm_rss":
            return int(float(subprocess.Popen(['sstat','-j',f'{os.environ.get("SLURM_JOB_ID")}.batch','-n','--format=MaxRSS'],stdout=subprocess.PIPE).communicate()[0].decode().replace("K","e3").replace("M","e6").replace("G","e9")))
        trc_mem, trc_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        if type == "trc_mem":
            return trc_mem
        if type == "trc_peak":
            return trc_peak
        return None

    def __init__(self):
        self.skip_count, self.skip_limit = None, None

    def init(self, D, maxN, cfg):
        self.skip_count = 0
        self.skip_limit = int(maxN * cfg.algo.build.skips)

    def has_train(self):
        pass

    def train(self, vecs):
        pass

    def add(self, vecs, start, count):
        """Manage build latency when adding samples"""
        self.skip_count = self.skip_count + count
        if self.skip_count <= self.skip_limit:
            return # Delay the add events until threshold is met
        batch_start = max(start + count - self.skip_count, 0)
        batch_count = min(self.skip_count, vecs.shape[0])
        # Add samples
        self.do_add(vecs, batch_start, batch_count)
        self.skip_count = 0

    def do_add(self, vecs, start, count):
        pass

    def update(self, vecs, start, count):
        """Manage build latency when updating samples"""
        self.skip_count = self.skip_count + count
        if self.skip_count <= 20 * self.skip_limit:
            return # Delay the update events until threshold is met
        batch_start = max(start + count - self.skip_count, 0)
        batch_count = min(self.skip_count, vecs.shape[0])
        # Apply updates
        self.do_update(vecs, batch_start, batch_count)
        self.skip_count = 0

    def do_update(self, vecs, start, count):
        self.do_add(vecs, start, count)

    def query(self, vecs, topk, cfg):
        pass

