from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset


class DyanmicsDataset(Dataset):
    def __init__(
        self,
        config,
        data_dict=Dict[str, Dict[str, Dict[str, str]]],
        is_train=False,
    ):
        self.config = config
        self.modalities = config["data"]["modalities"].keys()
        self.modality_file_list_data = {key: [] for key in self.modalities}
        self.timestamp_file_list_data = {key: [] for key in self.modalities}
        self.modality_processed_data = {key: [] for key in self.modalities}
        self.processed_state = None
        self.data_dict = data_dict
        self.is_train = is_train
        self.dt = self.config["data"]["dt"]
        self.horizon = self.config["data"]["horizon"]
        self.load_data()

    def load_data(self):
        for folder_name in self.data_dict:
            for modality in self.modalities:
                data_string = "data"
                filepath_string = "timestamp"
                data_path = self.data_dict[folder_name][modality][data_string]
                timestamp_path = self.data_dict[folder_name][modality][
                    filepath_string
                ]
                data = np.load(data_path)
                timestamps = np.loadtxt(timestamp_path, dtype=np.float64)
                timestamps = timestamps - timestamps[0]
                self.modality_file_list_data[modality].append(data)
                self.timestamp_file_list_data[modality].append(timestamps)

        # Todo: Do we need timestamps???
        for modality in self.modalities:
            samples_in_window = int(
                self.dt
                * self.config["data"]["modalities"][modality]["frequency"]
            )
            for single_file_data in self.modality_file_list_data[modality]:
                if len(single_file_data.shape) == 1:
                    single_file_data = single_file_data.reshape(-1, 1)
                n, k = single_file_data.shape
                reshaped_data = single_file_data[
                    : n - (n % samples_in_window)
                ].reshape(-1, samples_in_window, k)
                averaged_data = reshaped_data.mean(axis=1)
                self.modality_processed_data[modality].append(averaged_data)
            self.modality_processed_data[modality] = np.vstack(
                self.modality_processed_data[modality]
            )
        self.processed_state = self.construct_state()
        print(f"{self.processed_state.shape=}")

    def construct_state(self):
        base_state = self.modality_processed_data["super_odom"]  # (N, 13)
        # 4th 5th 6th 7th columns of base_state is quaternion convert it to rotation matrix for each sample
        rot_matrix = R.from_quat(base_state[:, 3:7]).as_matrix()
        # convert rotation matrix to 6D representation
        rot_6 = rot_matrix[:, :, [0, 1]].reshape(base_state.shape[0], -1)
        print(f"{rot_6.shape=}")
        steering_angle = self.modality_processed_data["cmd"][:, 1].reshape(
            -1, 1
        )  # (N, 1) instead of (N,)
        # tranform wheel angle to steering angle
        transformed_steering_angle = (
            steering_angle * (30.0 / 415.0) * (-np.pi / 180.0)
        )
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
        return (
            self.processed_state.shape[0] - self.horizon_rows - 1
        )  # no of datapoints for each modality is now the same

    def __getitem__(self, idx):
        current_state = self.processed_state[idx]
        environment = {}  # rgb etc?
        ground_truth = self.processed_state[
            idx + 1 : idx + 1 + (self.horizon_rows)
        ]
        action_horizon = None
        for modality in self.modalities:
            if (
                self.config["data"]["modalities"][modality]["type"]
                == "environment"
            ):
                environment[modality] = self.modality_processed_data[modality][
                    idx : idx + (self.horizon_rows)
                ]
            if self.config["data"]["modalities"][modality]["type"] == "action":
                action_horizon = self.modality_processed_data[modality][
                    idx : idx + (self.horizon_rows)
                ]
        return {
            "state": current_state,
            **environment,
            "action_horizon": action_horizon,
            "ground_truth": ground_truth,
        }

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.config["train"]["batch_size"],
            shuffle=self.is_train,
        )
