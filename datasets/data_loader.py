import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, SequentialSampler

from datasets.utils import split_and_subsample_batch
from datasets.ngsim_dataset import NGSIMDataset, NGSIMDatasetEval


def variable_time_collate_fn(batch, max_time_step):
    D = batch[0]["obs_data"].shape[-1]
    N = batch[0]["act_data"].shape[-1]  # number of labels

    combined_tt, inverse_indices = torch.unique(torch.cat([elem["time_steps"] for elem in batch]), sorted=True,
                                                return_inverse=True)

    offset = 0
    combined_obs = torch.zeros([len(batch), len(combined_tt), D])
    combined_mask = torch.zeros([len(batch), len(combined_tt), 1])
    combined_acts = torch.zeros([len(batch), len(combined_tt), N])

    for b, elem in enumerate(batch):

        indices = inverse_indices[offset:offset + len(elem["time_steps"])]
        offset += len(elem["time_steps"])

        combined_obs[b, indices] = elem["obs_data"]
        combined_mask[b, indices] = 1
        combined_acts[b, indices] = elem["act_data"]

    combined_tt = combined_tt.float()

    # if torch.max(combined_tt) != 0.:
    combined_tt = combined_tt / max_time_step #torch.max(combined_tt)

    data_dict = {
        "obs_data": combined_obs,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "act_data": combined_acts}

    data_dict = split_and_subsample_batch(data_dict)
    return data_dict


class NGSIMLoader(object):
    def __init__(self, cfg_data, dataset_file, mode='train'):
        self.cfg_data = cfg_data
        self.dataset_file = dataset_file
        if mode == 'train':
            self.ngsim_dataset = NGSIMDataset(cfg_data, dataset_file)
        else:
            self.ngsim_dataset = NGSIMDatasetEval(cfg_data, dataset_file)

    def get_test_dataloader(self):
        sampler = SequentialSampler(self.ngsim_dataset)
        return DataLoader(self.ngsim_dataset, batch_size=1, shuffle=False, sampler=sampler,
                          num_workers=4, pin_memory=True, collate_fn=lambda batch: batch[0])

    def split_train_test(self):
        test_ratio = self.cfg_data["test_ratio"]
        test_size = int(len(self.ngsim_dataset) * test_ratio)
        indices = np.arange(len(self.ngsim_dataset))
        if self.dataset_file != "trajdata_i101_trajectories-0750am-0805am.txt":
            np.random.shuffle(indices)
        test_sampler = SubsetRandomSampler(indices[:test_size])
        train_sampler = SubsetRandomSampler(indices[test_size:])
        # train_ngsim, test_ngsim = random_split(self.ngsim_dataset, [len(self.ngsim_dataset) - test_size, test_size])
        train_dataloader = DataLoader(self.ngsim_dataset, batch_size=self.cfg_data["batch_size"],
                                      sampler=train_sampler, num_workers=16,
                                      collate_fn=lambda batch: variable_time_collate_fn(batch, 1000), pin_memory=True)
        test_dataloader = DataLoader(self.ngsim_dataset, batch_size=1, sampler=test_sampler,
                                     num_workers=16, collate_fn=lambda batch: variable_time_collate_fn(batch, 1000), pin_memory=True)
        print(len(self.ngsim_dataset), len(train_dataloader), len(test_dataloader))

        return train_dataloader, test_dataloader


if __name__ == "__main__":
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    dataloader = NGSIMLoader(cfg.dataset, "trajdata_i101_trajectories-0750am-0805am.txt", mode="test").get_test_dataloader()
    for idx, data_dict in enumerate(dataloader):
        print(idx, data_dict["observed_data"].shape)




