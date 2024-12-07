from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2

from functools import cached_property

class DyanmicsDataset(Dataset):
    def __init__(
        self,
        config,
        data_dict=Dict[str, Dict[str, Dict[str, str]]],
        ds_type="train",
        normalization_stats=None,
    ):
        self.config = config
        self.modalities = config["data"]["modalities"].keys()
        self.data_dict = data_dict
        for folder in self.data_dict:
            image_data = self.data_dict[folder]["image_left_color"]["data"]
            image_data.sort(key=lambda x: int(x.split(".")[0]))
            self.data_dict[folder]["image_left_color"]["data"] = image_data

            height_data = self.data_dict[folder]["height_map_12x12"]["data"]
            height_data.sort(key=lambda x: int(x.split(".")[0]))
            self.data_dict[folder]["height_map_12x12"]["data"] = height_data

        self.is_train = ds_type == "train"
        self.dt = self.config["data"]["dt"]
        self.horizon = self.config["data"]["horizon_seconds"][ds_type]
        self.normalization_stats = normalization_stats
        self.file_order: List[Tuple[str, int]] = []
        self.file_cumsum: List[int] = []
        self.ds_type = ds_type
        self.load_data()

    def load_modality_from_folder(self, folder_name, modality, idx=-1):
        data_string = "data"
        filepath_string = "timestamp"
        timestamp_path = self.data_dict[folder_name][modality][filepath_string]
        base_path = Path(timestamp_path).parent
        timestamps = np.loadtxt(timestamp_path, dtype=np.float64)
        timestamps = timestamps - timestamps[0]
        if len(self.data_dict[folder_name][modality][data_string]) == 1:
            data_path = (
                base_path / self.data_dict[folder_name][modality][data_string][0]
            )
            return np.load(data_path)
        elif modality.startswith("height_map"):
            data_path = self.data_dict[folder_name][modality][data_string][idx]
            return np.load(base_path / data_path)
        elif modality.startswith("image"):
            data_path = self.data_dict[folder_name][modality][data_string][idx]
            return cv2.imread(str(base_path / data_path), cv2.IMREAD_UNCHANGED).astype(
                np.float64
            )

    def save_dict_numpy_to_npz(self, fname, dict_data):
        np.savez_compressed(fname, **dict_data)

    def load_dict_numpy_from_npz(self, fname):
        dict_data = np.load(fname, allow_pickle=True)
        return {key: dict_data[key] for key in dict_data}

    def save_array_to_npy(self, fname, array):
        np.save(fname, array)

    def load_array_from_npy(self, fname):
        return np.load(fname, allow_pickle=True)

    def load_data(self):
        for folder_name in self.data_dict:
            num_frames = (
                len(self.data_dict[folder_name]["height_map_12x12"]["data"])
                - self.horizon_rows
            )
            self.file_order.append((folder_name, num_frames))

        self.file_cumsum = np.cumsum([num_frames for _, num_frames in self.file_order])

        modality_processed_data = {key: [] for key in self.modalities}

        for modality in self.modalities:
            if modality.startswith("height_map") or modality.startswith("image"):
                continue

            samples_in_window = int(
                self.dt * self.config["data"]["modalities"][modality]["frequency"]
            )
            for folder_name in self.data_dict:
                single_file_data = self.load_modality_from_folder(folder_name, modality)
                if len(single_file_data.shape) == 1:
                    single_file_data = single_file_data.reshape(-1, 1)
                n, k = single_file_data.shape
                reshaped_data = single_file_data[: n - (n % samples_in_window)].reshape(
                    -1, samples_in_window, k
                )
                averaged_data = reshaped_data.mean(axis=1)
                modality_processed_data[modality].append(averaged_data)

        processed_state = []
        ground_truth = []
        envs = {}
        action_over_horizon = []

        for modality in self.modalities:
            if self.config["data"]["modalities"][modality]["type"] == "state":
                for instance_state, instance_cmd in zip(
                    modality_processed_data[modality],
                    modality_processed_data["cmd"],
                ):
                    # print(instance_state.shape, instance_cmd.shape)
                    processed_state_ = self.construct_state(
                        instance_state, instance_cmd
                    )
                    ground_truth_ = np.stack(
                        [
                            processed_state_[idx + 1 : idx + 1 + self.horizon_rows]
                            for idx in range(
                                processed_state_.shape[0] - self.horizon_rows - 1
                            )
                        ],
                    )
                    processed_state.append(processed_state_)
                    ground_truth.append(ground_truth_)
                processed_state = np.vstack(processed_state)
                ground_truth = np.concatenate(ground_truth)
            elif self.config["data"]["modalities"][modality]["type"] == "action":
                for instance_actions in modality_processed_data[modality]:
                    action_over_horizon_ = np.stack(
                        [
                            instance_actions[idx : idx + self.horizon_rows]
                            for idx in range(
                                instance_actions.shape[0] - self.horizon_rows
                            )
                        ],
                    )
                    action_over_horizon.append(action_over_horizon_)
                action_over_horizon = np.concatenate(action_over_horizon)
            elif self.config["data"]["modalities"][modality]["type"] == "environment":
                if modality.startswith("height_map") or modality.startswith("image"):
                    continue
                envs[modality] = np.concatenate(modality_processed_data[modality])

        self.save_array_to_npy(f"{self.ds_type}_processed_state.npy", processed_state)
        self.save_array_to_npy(f"{self.ds_type}_ground_truth.npy", ground_truth)
        self.save_array_to_npy(
            f"{self.ds_type}_action_over_horizon.npy", action_over_horizon
        )
        self.save_dict_numpy_to_npz(f"{self.ds_type}_envs.npz", envs)

        if self.is_train and self.normalization_stats is None:
            self.normalization_stats = self.get_normalization_stats()

        # self.normalize_data()

    @cached_property
    def processed_state(self):
        return self.load_array_from_npy(f"{self.ds_type}_processed_state.npy")

    @cached_property
    def ground_truth(self):
        return self.load_array_from_npy(f"{self.ds_type}_ground_truth.npy")

    @cached_property
    def action_over_horizon(self):
        return self.load_array_from_npy(f"{self.ds_type}_action_over_horizon.npy")

    @cached_property
    def envs(self):
        return self.load_dict_numpy_from_npz(f"{self.ds_type}_envs.npz")

    def get_normalization_stats(self):
        # check if "normalization_stats.npz" exists and load it
        if Path("normalization_stats.npz").exists():
            return self.load_dict_numpy_from_npz("normalization_stats.npz")

        # calculate mean and std for each column of processed state
        mean = np.mean(self.processed_state, axis=0)
        std = np.std(self.processed_state, axis=0)
        action_mean = np.mean(self.action_over_horizon)
        action_std = np.std(self.action_over_horizon)
        envs_mean = {}
        envs_var = {}

        if self.config["train"]["normalize_envs"]:
            # compute mean and std for env modalities via welford method
            for idx in range(self.__len__()):
                batch = self.__getitem__(idx)
                for modality in self.modalities:
                    if (
                        self.config["data"]["modalities"][modality]["type"]
                        == "environment"
                    ):
                        if modality.startswith("height_map") or modality.startswith(
                            "image"
                        ):
                            if self.config["train"]["normalize_images"]:
                                continue
                            # height_map and image are of the form (H, W, C), convert to (H*W, C)
                            batch[modality] = batch[modality].reshape(
                                -1, batch[modality].shape[-1]
                            )
                        if modality not in envs_mean:
                            envs_mean[modality] = np.zeros_like(batch[modality])
                            envs_var[modality] = np.zeros_like(batch[modality])
                        delta = batch[modality] - envs_mean[modality]
                        envs_mean[modality] += delta / (idx + 1)
                        delta2 = batch[modality] - envs_mean[modality]
                        envs_var[modality] += delta * delta2
        print("computed normalization stats")
        for modality in envs_mean:
            envs_var[modality] /= self.__len__()

        stats = {
            "mean": mean,
            "std": std,
            "action_mean": action_mean,
            "action_std": action_std,
            "envs_mean": envs_mean,
            "envs_var": envs_var,
        }
        self.save_dict_numpy_to_npz("normalization_stats.npz", stats)
        return stats

    def normalize_sample(self, sample):
        if self.normalization_stats is None:
            # print("Not normalizing data")
            return sample
        sample["state"] = (
            sample["state"] - self.normalization_stats["mean"]
        ) / self.normalization_stats["std"]
        sample["ground_truth"] = (
            sample["ground_truth"] - self.normalization_stats["mean"]
        ) / self.normalization_stats["std"]
        sample["action_horizon"] = (
            sample["action_horizon"] - self.normalization_stats["action_mean"]
        ) / self.normalization_stats["action_std"]

        if self.config["train"]["normalize_envs"]:
            for modality in self.normalization_stats["envs_mean"]:
                sample[modality] = (
                    sample[modality] - self.normalization_stats["envs_mean"][modality]
                ) / np.sqrt(self.normalization_stats["envs_var"][modality])
        return sample

    def construct_state(self, base_state, cmds):
        # base_state: (N, 13)
        # 4th 5th 6th 7th columns of base_state is quaternion convert it to rotation matrix for each sample
        rot_matrix = R.from_quat(base_state[:, 3:7]).as_matrix()
        # convert rotation matrix to 6D representation
        rot_6 = rot_matrix[:, :, [0, 1]].reshape(base_state.shape[0], -1)
        steering_angle = cmds[:, 1].reshape(-1, 1)  # (N, 1) instead of (N,)
        # tranform wheel angle to steering angle
        transformed_steering_angle = steering_angle * (30.0 / 415.0) * (-np.pi / 180.0)
        # create the transformed state by replacing columns 3 to 7 with rot_6 and appending the transformed_steering angle
        transformed_state = np.hstack(
            [
                base_state[:, :3],
                rot_6,
                base_state[:, 7:],
                transformed_steering_angle,
            ],
        )
        return transformed_state

    @property
    def horizon_rows(self):
        return np.ceil(self.horizon / self.dt).astype(int)

    def __len__(self):
        return self.ground_truth.shape[0]

    def __getitem__(self, idx):
        current_state = self.processed_state[idx]
        ground_truth = self.ground_truth[idx]
        action_horizon = self.action_over_horizon[idx]
        environment = {}
        # figure out folder name and index by binary search over self.file_cumsum
        folder_idx = np.searchsorted(self.file_cumsum, idx, side="right")
        start_idx = self.file_cumsum[folder_idx - 1] if folder_idx > 0 else 0
        folder_name, _ = self.file_order[folder_idx]
        frame_idx = idx - start_idx
        # print(f"{folder_name=}, {frame_idx=} {idx=} {start_idx=} {self.file_cumsum}")
        for modality in self.modalities:
            if self.config["data"]["modalities"][modality]["type"] == "environment":
                if modality.startswith("height_map") or modality.startswith("image"):
                    environment[modality] = self.load_modality_from_folder(
                        folder_name, modality, frame_idx
                    )
                else:
                    environment[modality] = self.envs[modality][idx]

        observation = {
            "state": current_state,
            **environment,
            "action_horizon": action_horizon,
            "ground_truth": ground_truth,
        }
        return self.normalize_sample(observation)

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.config["train"]["batch_size"],
            shuffle=self.is_train,
            num_workers=self.config["train"]["num_workers"],
            # prefetch_factor=self.config["train"]["prefetch_factor"],
        )
