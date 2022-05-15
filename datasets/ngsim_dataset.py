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
    def __init__(self, data_path, data_list, act_keys=('accel', 'turn_rate_global'), min_traj_length=50,
                 max_traj_length=None, normalize_data=True, act_low=-1.0, act_high=1.0, clip_std_multiple=np.inf):
        self.data_path = data_path
        self.data_list = data_list
        self.num_files = len(data_list)
        self.act_keys = act_keys
        self.min_traj_length = min_traj_length
        self.max_traj_length = max_traj_length
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
                    length = np.sum(np.sum(traj_data[i], axis=1) != 0, axis=0)
                    if length > min_traj_length:
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

        # time_steps = np.arange(0, observations.shape[1])
        time_steps = np.where(np.sum(np.abs(observations), axis=1) > 0)[0]
        n_sample_ts = int(np.random.rand() * len(time_steps)) + 2
        time_steps = time_steps[:n_sample_ts]

        observations = observations[time_steps, :]
        actions = observations[:, self.act_idxs]
        # length = np.where(np.sum(np.abs(observations), axis=1) == 0)[0][0]
        # observations = observations[:length]
        # if self.max_traj_length is not None:
        #     observations = observations[:self.max_traj_length]
        # actions = observations[:, self.act_idxs]
        # mask = np.sum(np.abs(observations), axis=1, keepdims=True) > 0
        # time_steps = np.arange(0, observations.shape[0], dtype=np.float32) / observations.shape[0]
        # observed_mask = np.repeat(mask.astype(np.float32), observations.shape[1], axis=1)
        # mask_predicted_data = np.repeat(mask.astype(np.float32), actions.shape[1], axis=1)

        if self.normalize_data:
            mean, std = self.data_statistics[file_id]
            observations = np.clip(observations, - std * self.clip_std_multiple, std * self.clip_std_multiple)
            observations = (observations - mean) / std
            actions = utils.normalize_range(actions, self.act_low, self.act_high)

        # return {"observed_data": np.expand_dims(observations, axis=0),
        #         "observed_tp": np.expand_dims(time_steps, axis=0),
        #         "data_to_predict": np.expand_dims(actions, axis=0),
        #         "tp_to_predict": np.expand_dims(time_steps, axis=0),
        #         "observed_mask": np.expand_dims(observed_mask, axis=0),
        #         "mask_predicted_data": np.expand_dims(mask_predicted_data, axis=0)}
        return {"obs_data": torch.from_numpy(observations),
                "time_steps": torch.from_numpy(time_steps),
                "act_data": torch.from_numpy(actions)}


if __name__ == '__main__':
    ngsim = NGSIMDataset("/home/khangtg/Documents/course/AI618_unsupervised_and_generative_models/code/ngsim_env/data/trajectories/ngsim.h5",
                         ['trajdata_i101_trajectories-0750am-0805am.txt'])
    print(ngsim.data_statistics)
    for key, val in ngsim[10].items():
        print(key, val.shape)
