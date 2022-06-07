import h5py
import numpy as np
import os
import torch

NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': '1',
    'trajdata_i101_trajectories-0805am-0820am.txt': '2',
    'trajdata_i101_trajectories-0820am-0835am.txt': '3',
    'trajdata_i80_trajectories-0400-0415.txt': '4',
    'trajdata_i80_trajectories-0500-0515.txt': '5',
    'trajdata_i80_trajectories-0515-0530.txt': '6'
}

'''
Common 
'''


def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def partition_list(lst, n):
    sublists = [[] for _ in range(n)]
    for i, v in enumerate(lst):
        sublists[i % n].append(v)
    return sublists


def str2bool(v):
    if v.lower() == 'true':
        return True
    return False


'''
data utilities
'''


def extract_mean_std(x):
    n_ids, n_steps, n_feats = x.shape
    # mean = np.mean(np.mean(x, axis=0, keepdims=True), axis=0, keepdims=True)
    # x = x - mean
    x_flatten = np.reshape(x, [-1, n_feats])
    mask = np.sum(np.abs(x_flatten), axis=1, keepdims=True) > 0
    n_valid_elems = np.sum(mask)
    mask = np.repeat(mask, n_feats, axis=1)
    mean = np.sum(x_flatten * mask, axis=0, keepdims=True) / n_valid_elems
    var = np.sum((x_flatten - mean)**2 * mask, axis=0, keepdims=True) / n_valid_elems
    std = np.sqrt(var) + 1e-8
    # mean = np.mean(x_flatten, axis=0, keepdims=True)
    # std = np.std(x_flatten, axis=0, keepdims=True) + 1e-8
    return mean, std


def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return np.array(lengths)


def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(x, axis=0, keepdims=True)
    x = x - mean
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)

    x = x / std
    return x, mean, std


def normalize_range(x, low, high):
    low = np.array(low).reshape(1, -1)
    high = np.array(high).reshape(1, -1)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
    return x


# def subsample_timepoints(data, time_steps, mask, n_tp_to_sample=None):
#     # n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
#     if n_tp_to_sample is None:
#         return data, time_steps, mask
#     n_tp_in_batch = len(time_steps)
#
#     if n_tp_to_sample > 1:
#         # Subsample exact number of points
#         assert (n_tp_to_sample <= n_tp_in_batch)
#         n_tp_to_sample = int(n_tp_to_sample)
#
#         for i in range(data.size(0)):
#             missing_idx = sorted(
#                 np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace=False))
#
#             data[i, missing_idx] = 0.
#             if mask is not None:
#                 mask[i, missing_idx] = 0.
#
#     elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
#         # Subsample percentage of points from each time series
#         percentage_tp_to_sample = n_tp_to_sample
#         for i in range(data.size(0)):
#             # take mask for current training sample and sum over all features -- figure out which time points don't have any measurements at all in this batch
#             current_mask = mask[i].sum(-1).cpu()
#             non_missing_tp = np.where(current_mask > 0)[0]
#             n_tp_current = len(non_missing_tp)
#             n_to_sample = int(n_tp_current * percentage_tp_to_sample)
#             subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace=False))
#             tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)
#
#             data[i, tp_to_set_to_zero] = 0.
#             if mask is not None:
#                 mask[i, tp_to_set_to_zero] = 0.
#
#     return data, time_steps, mask
#
#
# def cut_out_timepoints(data, time_steps, mask, n_points_to_cut=None):
#     # n_points_to_cut: number of consecutive time points to cut out
#     if n_points_to_cut is None:
#         return data, time_steps, mask
#     n_tp_in_batch = len(time_steps)
#
#     if n_points_to_cut < 1:
#         raise Exception("Number of time points to cut out must be > 1")
#
#     assert (n_points_to_cut <= n_tp_in_batch)
#     n_points_to_cut = int(n_points_to_cut)
#
#     for i in range(data.size(0)):
#         start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut - 5), replace=False)
#
#         data[i, start: (start + n_points_to_cut)] = 0.
#         if mask is not None:
#             mask[i, start: (start + n_points_to_cut)] = 0.
#
#     return data, time_steps, mask
#
#
def split_and_subsample_batch(data_dict, observed_ratio=None):
    # device = get_device(data_dict["data"])

    if observed_ratio is not None:
        n_observed_tp = int(data_dict["time_steps"].shape[0] * observed_ratio)
    else:
        n_observed_tp = data_dict["time_steps"].shape[0] - 1


    split_dict = {"observed_data": data_dict["obs_data"][:, :n_observed_tp, :],
                  "observed_tp": data_dict["time_steps"][:n_observed_tp],
                  "data_to_predict": data_dict["act_data"], #[:, n_observed_tp:, :],
                  "tp_to_predict": data_dict["time_steps"]} #[n_observed_tp:]}

    # split_dict["observed_mask"] = None
    # split_dict["mask_predicted_data"] = None
    split_dict["labels"] = None

    # if ("mask" in data_dict) and (data_dict["mask"] is not None):
    split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].repeat(1, 1, data_dict["obs_data"].shape[-1])
    split_dict["mask_predicted_data"] = data_dict["mask"].repeat(1, 1, data_dict["act_data"].shape[-1])

    # split_dict["mode"] = "extrap"
    return split_dict
