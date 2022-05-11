import numpy as np
import h5py
from torch.utils.data import Dataset


import utils


NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': '1',
    'trajdata_i101_trajectories-0805am-0820am.txt': '2',
    'trajdata_i101_trajectories-0820am-0835am.txt': '3',
    'trajdata_i80_trajectories-0400-0415.txt': '4',
    'trajdata_i80_trajectories-0500-0515.txt': '5',
    'trajdata_i80_trajectories-0515-0530.txt': '6'
}


class NGSIMDataset(Dataset):
    def __init__(self, data_path, data_list, act_keys=('accel', 'turn_rate_global'),
                 normalize_data=True, act_low=-1.0, act_high=1.0, clip_std_multiple=10.0):
        self.data_path = data_path
        self.data_list = data_list
        self.num_files = len(data_list)
        self.act_keys = act_keys
        self.normalize_data = normalize_data
        self.act_low = act_low
        self.act_high = act_high
        self.clip_std_multiple = clip_std_multiple

        self.database = h5py.File(data_path, 'r')
        self.feature_names = self.database.attrs["feature_names"]
        self.act_idxs = [i for (i, n) in enumerate(self.feature_names) if n in act_keys]

        self.mapping_idx = {}
        self.data_statistics = {}
        idx = 0
        for file_name in data_list:
            file_id = NGSIM_FILENAME_TO_ID[file_name]
            if file_id in self.database.keys():
                traj_data = self.database[file_id]
                self.data_statistics[file_id] = utils.extract_mean_std(traj_data) # save mean and std in each data file
                n_trajs = traj_data.shape[0]
                for i in range(n_trajs):
                    self.mapping_idx[idx] = file_id, i
                    idx += 1
            else:
                raise ValueError('invalid key to trajectory data: {}'.format(file_name))

    def __len__(self):
        return len(self.mapping_idx)

    def __getitem__(self, idx):
        file_id, traj_id = self.mapping_idx[idx]
        traj_data = self.database[file_id][traj_id]
        observations = np.array(traj_data, dtype=np.float32)
        # actions = observations[:, self.act_idxs]

        timesteps = np.where(np.sum(observations, axis=1) > 0)[0]
        obs = observations[timesteps, :]
        acts = obs[:, self.act_idxs]
        if self.normalize_data:
            mean, std = self.data_statistics[file_id]
            obs = np.clip(obs, - std * self.clip_std_multiple, std * self.clip_std_multiple)
            obs = (obs - mean) / std
            acts = utils.normalize_range(acts, self.act_low, self.act_high)

        return {"obs": np.expand_dims(obs, axis=0),
                "acts": np.expand_dims(acts, axis=0),
                "timesteps": np.expand_dims(timesteps, axis=0)}
