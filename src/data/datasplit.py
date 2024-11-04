import numpy as np
import os
import math
class DataSplitUtils:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.folder_names = [f for f in os.listdir(self.dataset_path)]     
        self.data_dict = dict.fromkeys(self.folder_names)
    def load_data(self, modalities):
        for folder_name in self.data_dict:
            self.data_dict[folder_name] = {}
        for dirs, _, files in os.walk(self.dataset_path):
            for file in files:
                if(file.endswith('.npy')):
                    path = os.path.join(dirs, file)
                    seperated_path = path.split('/')
                    if(seperated_path[-2] in modalities):                        
                        self.data_dict[seperated_path[1]][seperated_path[2]] = {}
                        self.data_dict[seperated_path[1]][seperated_path[2]]['data'] = os.path.abspath(path)
                        self.data_dict[seperated_path[1]][seperated_path[2]]['timestamp'] = os.path.abspath(os.path.join(dirs, 'timestamps.txt'))
        return self.data_dict

    def random_train_test_split(self, test_size):
        folders = list(self.data_dict.keys())
        np.random.shuffle(folders)
        split_index = math.floor(len(folders) * test_size)
        train_folders = folders[:-1*split_index]
        test_folders = folders[-1*split_index:]

        len_test_folders = len(test_folders)
        val_folders = test_folders[:len_test_folders//2]
        test_folders = test_folders[len_test_folders//2:]

        train_data = {key: self.data_dict[key] for key in train_folders}
        test_data = {key: self.data_dict[key] for key in test_folders}
        val_data = {key: self.data_dict[key] for key in val_folders}

        return train_data, test_data, val_data