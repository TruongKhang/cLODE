import torch

from datasets.utils import split_and_subsample_batch


def variable_time_collate_fn(batch):
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

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "obs_data": combined_obs,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "act_data": combined_acts}

    data_dict = split_and_subsample_batch(data_dict)
    return data_dict





