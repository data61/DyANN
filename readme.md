# Dynamic Approximate Nearest Neighbour Benchmarks

- [Dynamic Approximate Nearest Neighbour Benchmarks](#dynamic-approximate-nearest-neighbour-benchmarks)
  - [Install the framework](#install-the-framework)
  - [Running the code](#running-the-code)
  - [Adding new datasets](#adding-new-datasets)
  - [Adding new ANN algorithms](#adding-new-ann-algorithms)
  - [Benchmarks for Static ANN](#benchmarks-for-static-ann)

## Install the framework

Unpackage code and setup python environment
    > tar xvzf dyann.tar.gz
    > cd dyann
    > /apps/python/3.9.X/bin/python3 -m venv env
    > source env/bin/activate
    > pip install -r requirements.txt --no-cache-dir

## Running the code

Quick test
    > python download.py data=[datacol_quick]
    > python run.py data=[datacol_quick] algo=[linear,hnsw]
    > python plot-pareto.py data=[datacol_quick] algo=[linear,hnsw]
    > python plot-algo.py data=[datacol_quick] algo=[hnsw]

Preload all datasets and pregenerate all groundtruth (could take hours, ensure at least 30GB space)
    > python download.py data=[datacol,datacol_lerp,datacol_efreq,datacol_esfreq]
    > python download.py data=[featlearn,featlearn_lerp,featlearn_efreq,featlearn_esfreq]

Generate all benchmarking results (can easily take days or weeks, best run in parallel with a job scheduler)
    > python run.py data=[datacol,datacol_lerp,datacol_efreq,datacol_esfreq] algo=[linear,annoy,hnsw,ivfpq,scann,kdtree]
    > python run.py data=[featlearn,featlearn_lerp,featlearn_efreq,featlearn_esfreq] algo=[linear,annoy,hnsw,ivfpq,scann,kdtree]

## Adding new datasets

A template file for new datasets is provided at ./dyann/data/template.py

Usage Instructions:
1. Copy template.py and change the filename and class name for your new dataset
2. Update ./dyann/data/proxy.py to include the names you have chosen
3. Fill in each of the TODO items (refer to existing datasets for hints if needed)
4. Create any number of configuration sets in ./conf/data/
    with name property set to this filename
    and scale property providing an optional parameter sweep 

## Adding new ANN algorithms

A template file for new datasets is provided at ./dyann/algo/template.py

Usage Instructions:
1. Copy template.py and change the filename and class name for your new ANN algorithm
2. Update ./dyann/algo/proxy.py to include the names you have chosen
3. Fill in each of the TODO items (refer to existing algorithms for hints if needed)
4. Create both the build and search configuration files in ./conf/algo/
   with name property set to this filename
   the lists of parameters for the build and query properties will be swept

## Benchmarks for Static ANN

- https://github.com/erikbern/ann-benchmarks/
- https://github.com/matsui528/annbench