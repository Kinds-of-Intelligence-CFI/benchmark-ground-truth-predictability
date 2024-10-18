import argparse
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Callable, TypeVar

import numpy
import numpy as np
import pandas as pd
import yaml

from src.downloader_utils import ensure_file_downloaded


class DatasetPreparerAbstract(ABC):

    def __init__(self, num_few_shot=4):
        self.num_few_shot = num_few_shot

    def download_raw_data(
            self,
            target_path: str,
    ):
        # if no unpack, then add a filename with the right extension. Use the eval_id as the filename
        # get the extension from the source_url
        if isinstance(self.source_url, list):
            # check if self has attribute "subtasks" which is a list as well with the same length as self.source_url
            if not hasattr(self, "subtasks"):
                raise ValueError("If source_url is a list, then subtasks must be defined as well")
            # check if subtasks is a list:
            if not isinstance(self.subtasks, list):
                raise ValueError("If source_url is a list, then subtasks must be a list as well")
            if len(self.source_url) != len(self.subtasks):
                raise ValueError("If source_url is a list, then subtasks must have the same length")
            source_url = self.source_url
            subtasks = self.subtasks
        else:
            # wrap the source_url in a list
            # notice that this also works if there are multiple subtasks using the same data
            source_url = [self.source_url]
            subtasks = [None]
        for subtask, source_url in zip(subtasks, source_url):

            if not self.unpack:
                # find extension:
                # unless there is "format" towards the end of the string, the extension is the last part of the string
                if "format" not in source_url.split("/")[-1]:
                    extension = source_url.split(".")[-1]
                else:
                    # extension comes after format=
                    extension = source_url.split("format=")[-1]

                target_path_2 = target_path + "/" + self.eval_id
                if subtask is not None:
                    target_path_2 = target_path_2 + "_" + subtask

                target_path_2 += "." + extension
            else:
                target_path_2 = target_path

            ensure_file_downloaded(
                source_url=source_url,
                target_path=target_path_2,
                unpack=self.unpack,
                unpack_type=self.unpack_type,
            )

    @abstractmethod
    def transform_raw_data(self, path_to_raw_data, registry_data_path, rng_seed):
        """This function is specific to each dataset"""
        pass

    @staticmethod
    def create_chat_prompt_multiple_choice(sys_msg, question, choices, answers):
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)])
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def create_chat_prompt(sys_msg, question):
        user_prompt = f"{question}"
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt}
        ]

    @staticmethod
    def create_chat_example_multiple_choice(question, choices, answers, correct_answer):
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)])
        return [
            {"role": "system", "content": user_prompt, "name": "example_user"},
            {"role": "system", "content": correct_answer, "name": "example_assistant"},
        ]

    @staticmethod
    def create_chat_example(question, correct_answer):
        user_prompt = f"{question}"
        return [
            {"role": "system", "content": user_prompt, "name": "example_user"},
            {"role": "system", "content": correct_answer, "name": "example_assistant"},
        ]

    def register_dataset(self, registry_path):

        type = self.eval_template

        class_dict = {
            "match": "evals.elsuite.basic.match:Match",
            "includes": "evals.elsuite.basic.includes:Includes",
            "fuzzy_match": "evals.elsuite.basic.fuzzy_match:FuzzyMatch",
            "fact": "evals.elsuite.modelgraded.classify:ModelBasedClassify"
        }

        if type not in class_dict.keys():
            raise ValueError(f"Type {type} not recognized. Must be one of match, includes, fuzzy_match or fact")

        if not hasattr(self, "subtasks"):
            subtasks = [None]
        elif isinstance(self.subtasks, list):
            subtasks = self.subtasks
        else:
            subtasks = [self.subtasks]

        registry_yaml = {}

        for subtask in subtasks:
            eval_id = self.eval_id

            if subtask is not None:
                eval_id = eval_id + "_" + subtask

            registry_yaml[eval_id] = {
                "id": f"{eval_id}.test.v1",
                "metrics": ["accuracy"]
            }
            registry_yaml[f"{eval_id}.test.v1"] = {
                "class": class_dict[type],
                "args": {
                    "samples_jsonl": self._get_samples_path(subtask=subtask),
                }
            }

            if type == "fact":
                registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_type"] = "cot_classify"
                registry_yaml[f"{eval_id}.test.v1"]["args"]["modelgraded_spec"] = type
                if hasattr(self, "eval_completion_fn"):
                    registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_completion_fn"] = self.eval_completion_fn

            # now the few shot part:
            eval_id = eval_id + "_few_shot"

            registry_yaml[eval_id] = {
                "id": f"{eval_id}.test.v1",
                "metrics": ["accuracy"]
            }
            registry_yaml[f"{eval_id}.test.v1"] = {
                "class": class_dict[type],
                "args": {
                    "samples_jsonl": self._get_samples_path(subtask=subtask),
                    "few_shot_jsonl": self._get_samples_path(few_shot=True, subtask=subtask),
                    "num_few_shot": self.num_few_shot,
                }
            }

            if type == "fact":
                registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_type"] = "cot_classify"
                registry_yaml[f"{eval_id}.test.v1"]["args"]["modelgraded_spec"] = type
                if hasattr(self, "eval_completion_fn"):
                    registry_yaml[f"{eval_id}.test.v1"]["args"]["eval_completion_fn"] = self.eval_completion_fn

        self._save_yaml_registry(registry_path, registry_yaml)

        # this is actually not needed
        # if type == "fact":
        #     # then need to store the fact.yaml file in the modelgraded folder
        #     url = "https://github.com/openai/evals/raw/4b7a66bd45f06156656e021e170e7574f6cde3f5/evals/registry/modelgraded/fact.yaml"
        #     file_path = self._get_modelgraded_yaml_path(registry_path, type)
        #     # download the file in the right place
        #     ensure_file_downloaded(
        #         source_url=url,
        #         target_path=file_path,
        #         unpack=False,
        #         unpack_type=None,
        #     )

    def _get_samples_path(self, registry_path=None, subtask=None, few_shot=False):
        path_elements = [self.eval_id]

        if subtask is not None:
            path_elements += [subtask]

        if registry_path is not None:
            path_elements = [registry_path, "data"] + path_elements
            # create the folders if they do not exist
            os.makedirs(os.path.join(*path_elements), exist_ok=True)

        path_elements += ["samples.jsonl" if not few_shot else "few_shot.jsonl"]

        return os.path.join(*path_elements)

    def _get_yaml_path(self, registry_path):
        # create the folders if they do not exist
        os.makedirs(os.path.join(registry_path, "evals"), exist_ok=True)
        return os.path.join(registry_path, "evals", f"{self.eval_id}.yaml")

    @staticmethod
    def _get_modelgraded_yaml_path(registry_path, type):
        # create the folders if they do not exist
        os.makedirs(os.path.join(registry_path, "modelgraded"), exist_ok=True)
        return os.path.join(registry_path, "modelgraded", f"{type}.yaml")

    def _save_yaml_registry(self, registry_path, registry_yaml):
        with open(self._get_yaml_path(registry_path), "w") as f:
            yaml.dump(registry_yaml, f)


class DatasetPreparerFromURL(DatasetPreparerAbstract, ABC):
    source_url: str
    unpack: bool
    unpack_type: str
    eval_id: str
    eval_template: str

    @classmethod
    def __init_subclass__(cls):
        """This is run before the init of the subclass and raises an error if the class variables are not defined in the
        subclass"""
        required_class_variables = [
            'source_url',
            'unpack',
            'unpack_type',
            'eval_id',
            'eval_template',
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )
        # check if the eval_template is valid
        if cls.eval_template not in ["match", "includes", "fuzzy_match", "fact"]:
            raise ValueError(f"Eval template {cls.eval_template} not recognized. Must be one of match, includes, "
                             f"fuzzy_match or fact")


class DatasetPreparerBIGBench(DatasetPreparerAbstract, ABC):
    """This only work for json tasks which are multiple choice."""

    eval_id: str
    eval_template: str
    subtasks: str
    sys_msg: str
    ideal_index: bool
    unpack = False
    unpack_type = None
    allowed_tasks = [
        "formal_fallacies_syllogisms_negation",
        "logical_args",
        "crass_ai",
        "mnist_ascii",
        "geometric_shapes",
        "emoji_movie",
        "odd_one_out",
        "metaphor_boolean",
        "causal_judgment",
        "fantasy_reasoning",
        "moral_permissibility",
        "crash_blossom",
        "sentence_ambiguity",
        "dyck_languages",
        # the following are those with subtasks
        "intersect_geometry",
        "symbol_interpretation",
        "abstract_narrative_understanding",
        "conceptual_combinations",
        "cause_and_effect",
        "goal_step_wikihow",
        "arithmetic",
    ]

    @classmethod
    def __init_subclass__(cls):
        """This is run before the init of the subclass and raises an error if the class variables are not defined in the
        subclass"""
        required_class_variables = [
            'eval_id',
            'eval_template',
            'subtasks',
            'sys_msg',
            'ideal_index'
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )
        # check if the eval_template is valid
        if cls.eval_template not in ["match", "includes", "fuzzy_match", "fact"]:
            raise ValueError(f"Eval template {cls.eval_template} not recognized. Must be one of match, includes, "
                             f"fuzzy_match or fact")

        if cls.eval_id not in cls.allowed_tasks:
            raise ValueError(f"Eval id {cls.eval_id} not recognized. Must be one of {cls.allowed_tasks}")

    def __init__(self, num_few_shot=4):
        self.source_url = self._get_json_raw_url()
        super().__init__(num_few_shot=num_few_shot)

    def _get_json_raw_url(self, commit_n="6436ed17f979b138463224421f8e8977b89076ed"):
        urls = []
        if self.subtasks is not None:
            for subtask in self.subtasks:
                urls.append(
                    f"https://raw.githubusercontent.com/google/BIG-bench/{commit_n}/bigbench/benchmark_tasks/{self.eval_id}/{subtask}/task.json")
        else:
            urls = f"https://raw.githubusercontent.com/google/BIG-bench/{commit_n}/bigbench/benchmark_tasks/{self.eval_id}/task.json"
        return urls

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):

        rng = numpy.random.default_rng(rng_seed)
        rng2 = numpy.random.default_rng(rng_seed + 1)

        if isinstance(self.subtasks, list):
            subtasks = self.subtasks
        else:
            subtasks = [self.subtasks]

        for subtask in subtasks:
            if subtask is None:
                data_path = os.path.join(path_to_raw_data, f"{self.eval_id}.json")
            else:
                data_path = os.path.join(path_to_raw_data, f"{self.eval_id}_{subtask}.json")

            # load the dict from the json file
            with open(data_path, "r") as f:
                data = json.load(f)

            append_choices_to_input = True
            if "append_choices_to_input" in data.keys():
                append_choices_to_input = data["append_choices_to_input"]
            if hasattr(self, "append_choices_to_input"):
                # overwrite
                append_choices_to_input = self.append_choices_to_input

            if "task_prefix" in data.keys():
                if self.sys_msg == "":
                    sys_msg = data["task_prefix"]
                else:
                    sys_msg = data["task_prefix"].strip() + " " + self.sys_msg.strip()
            else:
                sys_msg = self.sys_msg

            # the samples are here:
            df = pd.DataFrame(data["examples"])

            # shuffle
            df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

            # some processing which is identical for all tasks

            # convert the target_scores (now a dict) to a list of tuples
            df["target_scores"] = df["target_scores"].apply(lambda x: list(x.items()))

            # shuffle
            df["target_scores"] = df["target_scores"].apply(lambda x: rng.permutation(x).tolist())

            # extract the choices from the target_scores
            df["choices"] = df["target_scores"].apply(lambda x: [y[0] for y in x])

            # extract the correct answer
            df["correct_answer"] = df["target_scores"].apply(
                lambda x: list([y[1] for y in x]).index(max([y[1] for y in x])))

            if self.ideal_index:
                df["ideal"] = df.apply(lambda x: str(x["correct_answer"] + 1), axis=1)
            else:
                df["ideal"] = df.apply(lambda x: str(x["choices"][x["correct_answer"]]), axis=1)

            # then do the task specific transformation
            df = self.task_specific_transformation(df)

            # now split into few shot and test
            if self.num_few_shot > 0:
                few_shot_df = df[:self.num_few_shot]
                # take those away from the df
                df = df[self.num_few_shot:]

                if append_choices_to_input:
                    # check if all "ideal" are the same in few_shot_df
                    if self.ideal_index and few_shot_df["ideal"].nunique() == 1:
                        # if so, we can permute the choices
                        for i, row in few_shot_df.iterrows():
                            # generate permutation indices of choices
                            perm = rng2.permutation(len(row["choices"]))
                            # apply the permutation to the choices
                            few_shot_df.at[i, "choices"] = [row["choices"][j] for j in perm]
                            # update the ideal index:
                            few_shot_df.at[i, "ideal"] = str(np.argwhere(perm == int(row["ideal"]) - 1)[0][0] + 1)

                    few_shot_df["sample"] = few_shot_df.apply(
                        lambda x: self.create_chat_example_multiple_choice(
                            x["input"],
                            (np.arange(len(x["choices"])) + 1).tolist(),
                            x['choices'], x["ideal"]), axis=1)
                else:
                    few_shot_df["sample"] = few_shot_df.apply(
                        lambda x: self.create_chat_example(
                            x["input"], x["ideal"]), axis=1)

                cols_to_save = ["sample"]

                # add the "comment" column if it exists
                if "comment" in few_shot_df.columns:
                    cols_to_save.append("comment")

                few_shot_df[cols_to_save].to_json(self._get_samples_path(registry_path, subtask=subtask, few_shot=True),
                                                  lines=True, orient="records")

            if append_choices_to_input:
                df["input"] = df.apply(
                    lambda x: self.create_chat_prompt_multiple_choice(sys_msg, f"{x['input']}",
                                                                      (np.arange(len(x["choices"])) + 1).tolist(),
                                                                      x['choices']), axis=1)
            else:
                df["input"] = df.apply(lambda x: self.create_chat_prompt(sys_msg, f"{x['input']}"), axis=1)

            cols_to_save = ["input", "ideal"]

            # add the "comment" column if it exists
            if "comment" in df.columns:
                cols_to_save.append("comment")

            df[cols_to_save].to_json(self._get_samples_path(registry_path, subtask=subtask), lines=True,
                                     orient="records")

    @staticmethod
    def task_specific_transformation(df):
        return df


# --- This part of code is adapted from the HELM repository: https://github.com/stanford-crfm/helm,
# which is released under Apache License 2.0 ---
DATASETS_DICT: Dict[str, DatasetPreparerAbstract] = {}
"""Dict of dataset names (or ids) to DatasetPreparerAbstract classes."""

F = TypeVar("F", bound=Callable[..., DatasetPreparerAbstract])


def dataset_class(name: str) -> Callable[[F], F]:
    """Register the run spec function under the given name."""

    def wrap(dataset: F) -> F:
        if name in DATASETS_DICT:
            raise ValueError(f"A dataset with name {name} already exists")
        DATASETS_DICT[name] = dataset
        return dataset

    return wrap


# --- End of adapted code ---

@dataset_class("neubaroco")
class NeuBAROCO(DatasetPreparerFromURL):
    source_url = "https://github.com/kmineshima/NeuBAROCO/raw/19d53af15967e23e11ba3e9eb159079eaed1e7bd/data/naloma2023/NeuBAROCO_NALOMA.tsv"
    unpack = False
    unpack_type = None
    eval_id = "neubaroco"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = ("I will provide you with a syllogism composed of two premises and a conclusion. You have to answer "
                   "whether the relation between premises and conclusion is entailment, neutral or contradiction. "
                   "Answer by only using the words 'entailment', 'neutral' or 'contradiction'.")

        data_path = os.path.join(path_to_raw_data, f"{self.eval_id}.tsv")

        df = pd.read_csv(data_path, sep="\t")

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            # few shot examples:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example(
                f"Premises: {x['premises_en']}\nConclusion: {x['hypothesis_en']}", x["gold"]), axis=1)
            few_shot_df[["sample", "figure", "conversion", "content-type", "atmosphere", "syllogism-type"]].to_json(
                self._get_samples_path(registry_path, few_shot=True), lines=True,
                orient="records")

        df["input"] = df.apply(
            lambda x: self.create_chat_prompt(sys_msg,
                                              f"Premises: {x['premises_en']}\nConclusion: {x['hypothesis_en']}"),
            axis=1)
        df["ideal"] = df["gold"]

        # save
        df[["input", "ideal", "figure", "conversion", "content-type", "atmosphere", "syllogism-type"]].to_json(
            self._get_samples_path(registry_path), lines=True, orient="records"
        )


@dataset_class("causal_judgment")
class CausalJudgment(DatasetPreparerBIGBench):
    eval_id = "causal_judgment"
    eval_template = "match"
    subtasks = None
    sys_msg = "Answer with 'Yes' or 'No'."
    ideal_index = False


@dataset_class("moral_permissibility")
class MoralPermissibility(DatasetPreparerBIGBench):
    eval_id = "moral_permissibility"
    eval_template = "match"
    subtasks = None
    sys_msg = "Answer with 'Yes' or 'No'."
    ideal_index = False


@dataset_class("commonsense_qa_2")
class CommonsenseQA2(DatasetPreparerFromURL):
    source_url = "https://raw.githubusercontent.com/LoryPack/LLM-LieDetector/91afc060155d6b0f30114dd003a0622ad8265006/data/raw_questions/commonsense_QA_v2_dev.json"
    # that is the same as https://github.com/allenai/csqa2/blob/362491ad71179e1aace820ef47b2b8d6315cc8ce/dataset/CSQA2_dev.json.gz but already decompressed
    unpack = False
    unpack_type = None
    eval_id = "commonsense_qa_2"
    eval_template = "match"

    def transform_raw_data(self, path_to_raw_data, registry_path, rng_seed=42):
        sys_msg = "I will provide you with a question about commonsense. Please answer with yes or no."

        data_path = os.path.join(path_to_raw_data, f"{self.eval_id}.json")

        df = pd.read_json(data_path, lines=True)

        # randomize the order of the df
        df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

        if self.num_few_shot > 0:
            few_shot_df = df[:self.num_few_shot]
            # take those away from the df
            df = df[self.num_few_shot:]

            few_shot_df["sample"] = few_shot_df.apply(lambda x: self.create_chat_example(
                x["question"], x["answer"]), axis=1)
            few_shot_df[["sample"]].to_json(self._get_samples_path(registry_path, few_shot=True), lines=True,
                                            orient="records")

        # Create test prompts and ideal completions
        df["input"] = df.apply(lambda x: self.create_chat_prompt(sys_msg, x["question"]), axis=1)
        df["ideal"] = df["answer"]

        # save input and ideal
        df[["input", "ideal"]].to_json(self._get_samples_path(registry_path), lines=True,
                                       orient="records")


if __name__ == "__main__":
    # set up arg parser that can list the datasets that you want to set up
    parser = argparse.ArgumentParser(description='Set up datasets')
    parser.add_argument('--datasets', nargs='+',
                        help='list of datasets to set up; if no dataset is provided, all datasets are downloaed',
                        choices=list(DATASETS_DICT.keys()), default=None)

    datasets = parser.parse_args().datasets
    if datasets is None:
        # then prepare all datasets:
        datasets = list(DATASETS_DICT.keys())

    for dataset_name in datasets:
        print(f"Downloading dataset {dataset_name}...")
        dataset_class = DATASETS_DICT[dataset_name]
        dataset = dataset_class()

        raw_data = os.path.join("raw_datasets", dataset_name)
        registry = os.path.join("registry")
        # create the folders if they do not exist
        os.makedirs(raw_data, exist_ok=True)
        os.makedirs(registry, exist_ok=True)

        dataset.download_raw_data(raw_data)
        print(f"Transforming dataset {dataset_name}...")
        dataset.transform_raw_data(raw_data, registry, rng_seed=42)
        print(f"Dataset {dataset_name} is set up.")
        dataset.register_dataset(registry)
