# Simple features predict LLM benchmark answers

Code for the paper "[Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers](https://arxiv.org/abs/2410.11672)".

# Citation
If you find our paper or code useful, please consider cite the following paper: 
```bibtex
@misc{pacchiardi2024leavingbarndooropen,
      title={Leaving the barn door open for {C}lever {H}ans: Simple features predict {LLM} benchmark answers}, 
      author={Lorenzo Pacchiardi and Marko Tesic and Lucy G. Cheke and José Hernández-Orallo},
      year={2024},
      eprint={2410.11672},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.11672}, 
}
```


# Reproducing the experiments:

## 1) Install dependencies
The original results were obtained with Python 3.9.19. The required dependencies can be installed with `pip install -r requirements.txt`. 

## 2) Download the data

To run the experiments, the raw performance data must be downloaded. The data comes from several sources; following the instruction below will download the data and put them in a folder `raw_results` in the root of this project.

### `legalbench`
The raw results for the considered scenarios of the `legalbench` dataset are available from Helm-Lite. Run the notebook `obtain_raw_results/download_helm/download_lite.ipynb` to download them.

### `CLadder`

Download the data from [this link](https://edmond.mpg.de/file.xhtml?fileId=246454&version=1.0) and decompress it into a folder `raw_results/cladder_output_files-1`. Released under [CC0 license](https://creativecommons.org/publicdomain/zero/1.0/)

### `ProntoQA`
The repository contains the result file obtained from ProntoQA for simplicity. The result file can be obtained by cloning [the original repository](https://github.com/asaparov/prontoqa/tree/227f5edb70c4c242565fff065d6873b588340f97) (released under Apache 2.0 license) and running the script `analyze_results.py`. 

### Datasets from the `KindsOfReasoning` collection
The following datasets are obtained from the [`KindsOfReasoning` collection](https://github.com/Kinds-of-Intelligence-CFI/KindsOfReasoning):
```
fantasy_reasoning, metaphor_boolean, anli, space_nli, wanli, babi_task_16, formal_fallacies_syllogisms_negation
```
See the above repository for credits and license information on those datasets.

To obtain them, download [this file](https://github.com/Kinds-of-Intelligence-CFI/KindsOfReasoning/blob/2e4e38f986decbd8716575d692cf8456bd52f824/full_processing_steps/2_results.tar.gz) and extract the folder in the `raw_results` folder.

### Remaining datasets

The evaluation results for the following datasets are included in the repository for convenience. They can be reproduced by following the instructions present in the `obtain_raw_results` folder.
```
neubaroco, moral_permissibility, causal_judgment, commonsense_qa_2 
```
Attributions:
- `moral_permissibility` and `causal_judgment` are obtained from [BIG-Bench](https://github.com/google/BIG-bench/) (License: Apache License 2.0).
- `commonsense_qa_2` can be obtained from [this repository](https://github.com/allenai/csqa2) (License: CC-BY-4.0).
- `neubaroco` can be obtained from [this repository](https://github.com/kmineshima/NeuBAROCO) (the file we used in the experiments is `data/naloma2023/NeuBAROCO_NALOMA.tsv`).

## 3) Run the experiments

Run the two notebooks `1_can_simple_features_predict_ground_truth.ipynb` and `2_effect_of_predicting_ground_truth_on_performance.ipynb` (in this order) in the folder `experiments` to reproduce all results, figures and tables present in the paper.


# Credits
- The code to download HELM-Lite was adapted from [this file](https://github.com/felipemaiapolo/efficbench/tree/master/generating_data/download_helm).
- The code to compute Word2Vec and FastText embeddings was adapted from https://github.com/lorypack/llm-liedetector (released under BSD-3-Clause license)
- We thank the creators of the [`NeuBAROCO`](https://github.com/kmineshima/NeuBAROCO) dataset to allow us to include their dataset and instance-level results on various LLMs in this repository.
