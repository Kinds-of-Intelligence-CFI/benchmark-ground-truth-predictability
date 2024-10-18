# Obtaining raw results

This folder contains the code needed to obtain the raw results for the `legalbench` (part of HELM-Lite) dataset and for `neubaroco, moral_permissibility, causal_judgment, commonsense_qa_2`. 

## `legalbench`

 Simply run the notebook `download_helm/download_lite.ipynb` to download them.

## `neubaroco, moral_permissibility, causal_judgment, commonsense_qa_2`

0. Install the required packages: set up a fresh virtual environment and run `pip install -r requirements.txt`.
1. Download the datasets and convert them in a common format by running `python set_up_datasets.py`. In particular, this will download the raw data into a folder named `raw_datasets`  and convert them to a format suitable for the [OpenAI `evals` library](https://github.com/openai/evals); the converted datasets and "registry" file are stored them in the `registry` folder. If you want to download one (or a few) datasets only, you can do so by running
    ```python 
    python set_up_datasets.py --datasets <dataset_name>
    ```
2. Run `./run_evals.sh` to run the LLMs on the datasets using `evals`. The results of the run will be stored in `../raw_results`. The provided version of `run_evals.sh` runs one LLM only. Before running `run_evals.sh`, you need to modify it by specifying the correct path. Moreover, running that uses the OpenAI API and thus requires specifying your OpenAI API key in `.env`. Notice that some of the results (attached to this repository) were obtained with LLMs that are not available any longer.
   