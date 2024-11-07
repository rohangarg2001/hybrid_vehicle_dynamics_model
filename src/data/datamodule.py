import torch
from torch.utils.data import Dataset
from typing import Dict
import numpy as np


class DyanmicsDataset(Dataset):
    def __init__(
        self,
        config,
        data_dict=Dict[str, Dict[str, Dict[str, str]]],     ## data_dict is like instance_name then the modality name and then file name for data amd timestamp
        is_train=False,
    ):
        self.config=config                      # config file
        self.modalities=config["data"]["modalities"].keys()             # all the modalities
        self.modality_file_list_data = {key:[] for key in self.modalities}      
        self.timestamp_file_list_data = {key:[] for key in self.modalities}
        self.modality_processed_data = {key:[] for key in self.modalities}
        self.data_dict = data_dict
        self.is_train = is_train
        self.dt = self.config['data']['dt']
        self.horizon = self.config['data']['horizon']
        self.load_data()

    def load_data(self):
        # Using data_dict, for each file path, go over all modalities and for each modality
        # load the 'data' filepath into memory via np and the corresponding timestamp data
        # now for a given filepath and modality generate one iterate of data
        # save each modality in an list of numpy objects
        for folder_name in self.data_dict:
            for modality in self.modalities:
                data_string = 'data'
                filepath_string = 'filepath'
                data_path = self.data_dict[folder_name][modality][data_string]
                timestamp_path = self.data_dict[folder_name][modality][filepath_string]                
                data = np.load(data_path)
                timestamps = np.loadtxt(timestamp_path, dtype=np.float64)
                timestamps = timestamps - timestamps[0]
                self.modality_file_list_data[modality].append(data)
                self.timestamp_file_list_data[modality].append(timestamps)

        # Todo: Do we need timestamps???
        for modality in self.modalities:
            samples_in_window = int(self.dt*self.config["data"]["modalities"][modality]["frequency"])     ## how many datapoints for different modalities in dt time
            for single_file_data in self.modality_file_list_data[modality]:         #for a modality single_file_data would be data of that modality and instance
                n, k = single_file_data.shape                                       # num_datapints, dim
                ## REMOVING THE EXTRA DATA POINTS from the end of the data 
                reshaped_data = single_file_data[:n - (n % samples_in_window)].reshape(-1, samples_in_window, k)            # removing the extra timesteps and making it in an array
                averaged_data = reshaped_data.mean(axis=1)                                                                  # in dt time we take the average value of the data
                self.modality_processed_data[modality].append(averaged_data)                                                # array of modality data for the entire instance
            self.modality_processed_data[modality] = np.vstack(self.modality_processed_data[modality])                      # combining array of modality for all instances.

    # def form_input_output(self):
    #     self.
    #     for modality in self.modalities:


    def __len__(self):
        return (list(self.modality_processed_data.values())[0].shape[0] - self.horizon/self.dt - 1)                             ## no of datapoints for each modality is now the same

    def __getitem__(self, idx):
        # return idx row from each modality in self.modality_processed_data, so idx is the index at take the current state
        
        current_state = {}
        environment = {}                ## rgb etc?
        ground_truth = {}
        for modality in self.modalities:
            if(self.config['data']['modalities'][modality]['type'] == 'state'):
                current_state[modality] = self.modality_processed_data[modality][idx]
                ground_truth[modality] = self.modality_processed_data[modality][idx+1 : idx+1 + (self.horizon / self.dt)]
            if(self.config['data']['modalities'][modality]['type'] == 'action'):
                action_horizon = self.modality_processed_data[modality][idx : idx + (self.horizon / self.dt)]
        return current_state, environment, action_horizon, ground_truth
