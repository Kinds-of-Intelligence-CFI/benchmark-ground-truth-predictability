import os

import pandas as pd

from results_loaders import CLadder, ProntoQA


def load_with_conditions(filename, overwrite_res=False):
    if not os.path.exists(filename) or overwrite_res:
        print("File not found or overwrite requested. Creating new dataframe.")
        df = pd.DataFrame()
    elif filename.split(".")[-1] == "csv":
        print("Loading existing dataframe.")
        df = pd.read_csv(filename)
    elif filename.split(".")[-1] == "pkl":
        print("Loading existing dataframe.")
        df = pd.read_pickle(filename)
    else:
        raise ValueError("File format not recognized. Please use .csv or .pkl.")

    return df


def save_dataframe(filename, res_df):
    if filename.endswith(".csv"):
        res_df.to_csv(filename, index=False)
    elif filename.endswith(".pkl"):
        res_df.to_pickle(filename)
    elif filename.endswith(".json"):
        res_df.to_json(filename)
    else:
        raise ValueError("filename not recognized")


def initialize_instance(dataset):
    llms = None
    ideal_col_name = "ground_truth"

    if len(dataset) == 3:
        dataset_class, scenario, subscenario = dataset
        instance = dataset_class(scenario=scenario, subscenario=subscenario, verbose=False, llms=llms,
                                 base_path_raw="../raw_results/helm_lite_v1.0.0")
        dataset_name = f"{scenario}_{subscenario}"
        group = "HELM-Lite"
    elif len(dataset) == 2:
        dataset_class, eval = dataset
        instance = dataset_class(task=eval, verbose=False, llms=llms, base_path_raw="../raw_results/")
        dataset_name = eval
        group = "eval"
    else:
        dataset_class = dataset[0]
        instance = dataset_class(verbose=False, base_path_raw="../raw_results/")
        if isinstance(instance, CLadder):
            ideal_col_name = "truth_norm"
            dataset_name = "CLadder"
        elif isinstance(instance, ProntoQA):
            ideal_col_name = "expected_label"
            dataset_name = "ProntoQA"
        group = "others"

    return instance, dataset_name, ideal_col_name, group


def Cohen_correction(accuracy, random_guess):
    return (accuracy - random_guess) / (1 - random_guess)
