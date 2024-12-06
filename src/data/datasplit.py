import json
import math
import os
from pathlib import Path
from typing import List
import numpy as np


class DataSplitUtils:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.folder_names = [f for f in os.listdir(self.dataset_path)]
        self.data_dict = dict.fromkeys(self.folder_names)

    def load_data(self, modalities):
        """Loads data from the dataset and organizes it into a dictionary.
        This function traverses the dataset directory, identifies files with a '.npy' extension,
        and organizes them into a dictionary based on the specified modalities. Each entry in the
        dictionary contains the absolute path to the data file and the corresponding timestamp file.

        Args:
            modalities (list of str): List of modalities to be included in the dataset.

        Returns:
            dict: A dictionary where the key is the instance name and the value is another dictionary
                  containing the data file path and timestamp file path.
        """
        for folder_name in self.data_dict:
            self.data_dict[folder_name] = {}

        base_path = Path(self.dataset_path)

        for trajectory_instance in base_path.iterdir():
            if not trajectory_instance.is_dir():
                continue
            print(f"Processing {trajectory_instance.name}")
            for modality in modalities:
                modality_path = trajectory_instance / modality
                timestamp_path = modality_path / "timestamps.txt"
                if not modality_path.is_dir():
                    continue
                if not timestamp_path.is_file():
                    continue
                print(f"\tProcessing {modality_path}")
                data_files = [
                    f.name
                    for f in modality_path.iterdir()
                    if f.name != "timestamps.txt" and f.is_file()
                ]
                self.data_dict[trajectory_instance.name][modality] = {
                    "data": data_files,
                    "timestamp": str(timestamp_path),
                }

        return self.data_dict

    def random_train_test_split(self, test_size, path: str = "data_split.json"):
        """
        Splits the dataset into random train, test, and validation sets,
        saves to a json file, and returns the splits.
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
                               Should be between 0.0 and 1.0.
        Returns:
            tuple: A tuple containing three dictionaries:
                - train_data (dict): Training data split.
                - test_data (dict): Testing data split.
                - val_data (dict): Validation data split.
        """

        folders = list(self.data_dict.keys())
        np.random.shuffle(folders)
        split_index = math.floor(len(folders) * test_size)
        train_folders = folders[: -1 * split_index]
        test_folders = folders[-1 * split_index :]

        len_test_folders = len(test_folders)
        val_folders = test_folders[len_test_folders // 2 :]
        test_folders = test_folders[: len_test_folders // 2]

        train_data = {key: self.data_dict[key] for key in train_folders}
        test_data = {key: self.data_dict[key] for key in test_folders}
        val_data = {key: self.data_dict[key] for key in val_folders}

        # save in a json file
        with open(path, "w") as fp:
            json.dump({"train": train_data, "test": test_data, "val": val_data}, fp)

        return train_data, test_data, val_data

    def load_random_train_test_split(self, path: str = "data_split.json"):
        with open(path, "r") as f:
            data = json.load(f)
        return data["train"], data["test"], data["val"]

    def rebalance_and_filter_split(
        self,
        path: str,
        modalities: List[str],
        old_split: str = "data_split_full_raw.json",
        test_size: float = 0.2,
    ):
        with open(old_split, "r") as f:
            raw_data_split = json.load(f)
        raw_data_total = {
            **raw_data_split["train"],
            **raw_data_split["test"],
            **raw_data_split["val"],
        }
        # filter out entries which dont have all modalities
        filtered_data = {}
        for traj, data in raw_data_total.items():
            if all(modality in data.keys() for modality in modalities):
                filtered_data[traj] = data
        print(f"Filtered out {len(raw_data_total) - len(filtered_data)} trajectories")
        # length of filtered_data[traj]["data"] is the length of the trajectory
        # use this length to split into train test val sets, each traj should be part of only one set

        traj_list = list(filtered_data.keys())
        np.random.shuffle(traj_list)
        train_data = {}
        test_data = {}
        val_data = {}
        total_length = 0
        for traj in traj_list:
            total_length += len(filtered_data[traj]["height_map_12x12"]["data"])
        cur_train_length = 0
        cur_val_length = 0
        cur_test_length = 0
        max_train_length = total_length * (1 - test_size)
        max_val_length = total_length * test_size / 2
        for i, traj in enumerate(traj_list):
            traj_length = len(filtered_data[traj]["height_map_12x12"]["data"])
            if cur_train_length + traj_length <= max_train_length:
                train_data[traj] = filtered_data[traj]
                cur_train_length += traj_length
            elif cur_val_length + traj_length <= max_val_length:
                val_data[traj] = filtered_data[traj]
                cur_val_length += traj_length
            else:
                test_data[traj] = filtered_data[traj]
                cur_test_length += traj_length
        print(f"Train set length: {cur_train_length}")
        print(f"Val set length: {cur_val_length}")
        print(f"Test set length: {cur_test_length}")
        print(f"Total length: {total_length}")
        print(f"Number of train trajectories: {len(train_data)}")
        print(f"Number of val trajectories: {len(val_data)}")
        print(f"Number of test trajectories: {len(test_data)}")
        with open(path, "w") as fp:
            json.dump({"train": train_data, "test": test_data, "val": val_data}, fp)
