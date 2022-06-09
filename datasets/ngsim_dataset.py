import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


import datasets.utils as utils


NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': '1',
    'trajdata_i101_trajectories-0805am-0820am.txt': '2',
    'trajdata_i101_trajectories-0820am-0835am.txt': '3',
    'trajdata_i80_trajectories-0400-0415.txt': '4',
    'trajdata_i80_trajectories-0500-0515.txt': '5',
    'trajdata_i80_trajectories-0515-0530.txt': '6'
}


class NGSIMDataset(Dataset):
    def __init__(self, config, dataset_file, mode='train'):
        self.config = config
        self.data_path = config["data_path"]
        self.dataset_file = dataset_file
        self.mode = mode
        self.act_keys = config["act_keys"]
        self.min_traj_length = config["min_traj_length"]
        # self.max_traj_length = max_traj_length
        self.normalize_data = config["normalize_data"]
        self.act_low = np.array(config["act_low"], dtype=np.float32)
        self.act_high = np.array(config["act_high"], dtype=np.float32)
        self.clip_std_multiple = config["clip_std_multiple"]
        self.max_obs_length = config["max_obs_length"]

        self.database = h5py.File(self.data_path, 'r')
        self.feature_names = self.database.attrs["feature_names"]
        # self.input_dims = len(self.feature_names)
        self.act_idxs = [i for (i, n) in enumerate(self.feature_names) if n in self.act_keys]

        self.mapping_idx = {}
        self.data_statistics = {}
        idx = 0
        # for file_name in self.data_list:
        file_id = NGSIM_FILENAME_TO_ID[dataset_file]
        if file_id in self.database.keys():
            self.all_traj_data = np.array(self.database[file_id], dtype=np.float32)
            mean, std = utils.extract_mean_std(self.all_traj_data)  # save mean and std in each data file
            mean[:, self.act_idxs] = (self.act_low + self.act_high) / 2
            std[:, self.act_idxs] = (self.act_high - self.act_low) / 2
            self.data_statistics = mean, std
            n_trajs = self.all_traj_data.shape[0]
            for i in range(n_trajs):
                length = np.sum(np.sum(np.abs(self.all_traj_data[i]), axis=1) != 0, axis=0)
                if length > self.min_traj_length:
                    self.mapping_idx[idx] = i #file_id, i
                    idx += 1
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(self.dataset_file))

    def __len__(self):
        return len(self.mapping_idx)

    def __getitem__(self, idx):
        observations = self.all_traj_data[self.mapping_idx[idx]]

        time_steps = np.where(np.sum(np.abs(observations), axis=1) > 0)[0]
        n_sample_ts = int(np.random.rand() * len(time_steps)) + 4 if self.max_obs_length is not None else self.max_obs_length
        time_steps = time_steps[:n_sample_ts]

        observations = observations[time_steps, :]
        actions = observations[:, self.act_idxs]

        if self.normalize_data:
            mean, std = self.data_statistics
            assert mean.shape[-1] == observations.shape[-1]
            assert std.shape[-1] == observations.shape[-1]
            observations = np.clip(observations, - std * self.clip_std_multiple, std * self.clip_std_multiple)
            observations = (observations - mean) / std
            actions = observations[:, self.act_idxs]
            actions = np.clip(actions, -1, 1)
            observations[:, self.act_idxs] = actions

        return {"obs_data": torch.from_numpy(observations).float(),
                "time_steps": torch.from_numpy(time_steps),
                "act_data": torch.from_numpy(actions).float()}


class NGSIMDatasetSim(Dataset):
    def __init__(self, config, dataset_file, use_multi_agents=True):
        self.config = config
        self.train_data_path = config["data_path"]
        self.test_data_path = config["test_data_path"]
        self.dataset_file = dataset_file
        self.act_keys = config["act_keys"]
        self.max_obs_length = config["max_obs_length"]
        self.normalize_data = config["normalize_data"]
        # self.act_low = config["act_low"]
        # self.act_high = config["act_high"]
        self.clip_std_multiple = config["clip_std_multiple"]

        self.train_db = h5py.File(self.train_data_path, 'r')
        self.test_db = h5py.File(self.test_data_path, 'r')

        self.data_statistics = {}
        file_id = NGSIM_FILENAME_TO_ID[dataset_file]

        self.all_train_data = np.array(self.train_db[file_id], dtype=np.float32)
        feature_names = self.train_db.attrs["feature_names"]
        # self.input_dims = len(self.feature_names)
        self.act_idxs = [i for (i, n) in enumerate(feature_names) if n in self.act_keys]
        mean, std = utils.extract_mean_std(self.all_train_data)  # save mean and std in each data file
        act_low = np.array(config["act_low"], dtype=np.float32)
        act_high = np.array(config["act_high"], dtype=np.float32)
        mean[:, self.act_idxs] = (act_low + act_high) / 2
        std[:, self.act_idxs] = (act_high - act_low) / 2
        self.data_statistics = mean, std

        self.test_data = np.array(self.test_db[file_id], dtype=np.float32)
        if self.max_obs_length is not None:
            self.test_data = self.test_data[:, :self.max_obs_length]
        if not use_multi_agents:
            selected_ids = np.random.choice(self.test_data.shape[0], 1)
            self.test_data = self.test_data[selected_ids]

        if self.normalize_data:
            # mean, std = self.data_statistics
            assert mean.shape[-1] == self.test_data.shape[-1]
            assert std.shape[-1] == self.test_data.shape[-1]
            mean, std = np.expand_dims(mean, axis=0), np.expand_dims(std, axis=0)
            observations = np.clip(self.test_data, - std * self.clip_std_multiple, std * self.clip_std_multiple)
            self.test_data = (observations - mean) / std
            actions = self.test_data[:, :, self.act_idxs]
            actions = np.clip(actions, -1, 1)
            self.test_data[:, :, self.act_idxs] = actions

    def __len__(self):
        return self.test_data.shape[1] - 1

    # def __getitem__(self, idx):
    #     observations = self.test_data[:, idx:(idx+1)]
    #
    #     time_steps = np.arange(observations.shape[1]+200, dtype=np.float32) / 250 #/ self.test_data.shape[1]
    #
    #
    #
    #     data_dict = {"observed_data": torch.from_numpy(observations).float(),
    #                  "time_steps": torch.from_numpy(time_steps).float()}
    #                  # "observed_tp": torch.from_numpy(time_steps[:n_observed_tp]).float(),
    #                  # "tp_to_predict": torch.from_numpy(time_steps[n_observed_tp:]).float()}
    #
    #     # data_dict["observed_mask"] = torch.ones_like(data_dict["observed_data"])
    #
    #     return data_dict


if __name__ == '__main__':
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    ngsim = NGSIMDatasetSim(cfg.dataset, "trajdata_i101_trajectories-0750am-0805am.txt")
    print(ngsim.data_statistics)
    for idx in range(len(ngsim)):
        print(idx, ngsim[idx]["observed_data"].shape)
