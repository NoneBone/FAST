# FAST

## Overview

This repo is the open-sourced code for our work *FAST: A Holistic Framework for Optimizing Memory-I/O, Computation, and Sampling in Temporal GNN Training*.

## Requirements

Please ensure the following dependencies are installed manually. For other Python dependencies, refer to `requirements.txt`.

- python >= 3.6.13
- pytorch >= 1.8.1
- dgl >= 0.9.1
- CUDA toolkit >= 11.1
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

The Core component is implemented using C++ and CUDA, please compile *fast* with the following command:
> bash ./buildFast.sh

## Datasets
We employ four publicly available datasets in our experiments: **LastFM**, **Wiki-Talk**, **Bitcoin**, and **GDELT**.

*   The **LastFM** and **GDELT** datasets can be downloaded automatically via the provided script `down.sh`.
*   The **Wiki-Talk** dataset is available at [http://snap.stanford.edu/data/wiki-talk-temporal.html](http://snap.stanford.edu/data/wiki-talk-temporal.html).
*   The **Bitcoin** dataset is available at [https://networkrepository.com/soc-bitcoin.php](https://networkrepository.com/soc-bitcoin.php).

**Note:** After downloading the raw Wiki-Talk and Bitcoin data, a preprocessing step is required to convert them into the temporal graph format used in this work. Please run the following commands sequentially:

```sh
python fast_util/preprocess/txt2csv.py --data dataset_name --txt graph_data_source_file
python fast_util/preprocess/gen_graph.py --data dataset_name 
```

## Train

To train a model, specify the target dataset and model. For example, the command below trains TGAT on the Wiki-Talk dataset, with its configuration file at `config/TGAT.yml`.
```sh
python train.py --data WIKITALK --model_name TGAT
```

(Note: The `TGAT.yml` configuration file is identical to the one used in the open-source framework TGL. For detailed parameter definitions, please refer to `config/readme.yml`.)
