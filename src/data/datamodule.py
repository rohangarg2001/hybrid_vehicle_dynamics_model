import torch
from torch.utils.data import Dataset
from typing import Dict
import numpy as np


class DyanmicsDataset(Dataset):
    def __init__(
        self,
        config,
        data_dict=Dict[str, Dict[str, Dict[str, str]]],
        is_train=False,
    ):
        self.config=config
        self.modalities=config["data"]["modalities"].keys()
        self.modality_file_list_data = {key:[] for key in self.modalities}
        self.timestamp_file_list_data = {key:[] for key in self.modalities}
        self.modality_processed_data = {key:[] for key in self.modalities}
        self.data_dict = data_dict
        self.is_train = is_train
        self.load_data()

    def load_data(self):
        # Using data_dict, for each file path, go over all modalities and for each modality
        # load the 'data' filepath into memory via np and the corresponding timestamp data
        # now for a given filepath and modality generate one iterate of data
        # save each modality in an list of numpy objects
        for folder_name in self.data_dict:
            for modality in self.modalities:
                data_path = self.data_dict[folder_name][modality]["data"]
                timestamp_path = self.data_dict[folder_name][modality]['timestamp']
                
                data = np.load(data_path)
                timestamps = np.loadtxt(timestamp_path, dtype=np.float64)
                timestamps = timestamps - timestamps[0]
                self.modality_file_list_data[modality].append(data)
                self.timestamp_file_list_data[modality].append(timestamps)

        # Todo: Do we need timestamps???
        for modality in self.modalities:
            samples_in_window = int(self.config["data"]["dt"]*self.config["data"]["modalities"][modality]["frequency"])
            for single_file_data in self.modality_file_list_data[modality]:
                n, k = single_file_data.shape
                reshaped_data = single_file_data[:n - (n % samples_in_window)].reshape(-1, samples_in_window, k)
                averaged_data = reshaped_data.mean(axis=1)
                self.modality_processed_data[modality].append(averaged_data)
            self.modality_processed_data[modality] = np.vstack(self.modality_processed_data[modality])


    def __len__(self):
        return list(self.modality_processed_data.values())[0].shape[0]

    def __getitem__(self, idx):
        # return idx row from each modality in self.modality_processed_data
        # return [idx:idx+horizon*dt] for ground_truth
        # drop last few rows for all modalities as there wouldnt be enough horizon for them
        return {modality: data[idx] for modality, data in self.modality_processed_data.items()}
