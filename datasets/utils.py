import h5py
import numpy as np
import os


'''
Const
'''
NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': 1,
    'trajdata_i101_trajectories-0805am-0820am.txt': 2,
    'trajdata_i101_trajectories-0820am-0835am.txt': 3,
    'trajdata_i80_trajectories-0400-0415.txt': 4,
    'trajdata_i80_trajectories-0500-0515.txt': 5,
    'trajdata_i80_trajectories-0515-0530.txt': 6
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


def write_trajectories(filepath, trajs):
    np.savez(filepath, trajs=trajs)


def load_trajectories(filepath):
    return np.load(filepath)['trajs']


def filename2label(fn):
    s = fn.find('-') + 1
    e = fn.rfind('_')
    return fn[s:e]


def load_trajs_labels(directory, files_to_use=[0, 1, 2, 3, 4, 5]):
    filenames = [
        'trajdata_i101_trajectories-0750am-0805am_trajectories.npz',
        'trajdata_i101_trajectories-0805am-0820am_trajectories.npz',
        'trajdata_i101_trajectories-0820am-0835am_trajectories.npz',
        'trajdata_i80_trajectories-0400-0415_trajectories.npz',
        'trajdata_i80_trajectories-0500-0515_trajectories.npz',
        'trajdata_i80_trajectories-0515-0530_trajectories.npz'
    ]
    filenames = [filenames[i] for i in files_to_use]
    labels = [filename2label(fn) for fn in filenames]
    filepaths = [os.path.join(directory, fn) for fn in filenames]
    trajs = [load_trajectories(fp) for fp in filepaths]
    return trajs, labels


'''
data utilities
'''


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
    low = np.array(low)
    high = np.array(high)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
    return x


def load_x_feature_names(filepath, ngsim_filename):
    f = h5py.File(filepath, 'r')
    xs = []
    traj_id = NGSIM_FILENAME_TO_ID[ngsim_filename]
    # in case this nees to allow for multiple files in the future
    traj_ids = [traj_id]
    for i in traj_ids:
        if str(i) in f.keys():
            xs.append(f[str(i)])
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    x = np.concatenate(xs)
    feature_names = f.attrs['feature_names']
    return x, feature_names


def load_data(
        filepath,
        act_keys=['accel', 'turn_rate_global'],
        ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt',
        debug_size=None,
        min_length=50,
        normalize_data=True,
        shuffle=False,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf):
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names(filepath, ngsim_filename)
    print(x.shape)
    print(feature_names)

    # optionally keep it to a reasonable size
    if debug_size is not None:
        x = x[:debug_size]

    if shuffle:
        idxs = np.random.permutation(len(x))
        x = x[idxs]

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)
    print("Lengths: ", len(lengths))

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i, :l])
    x = np.concatenate(xs)
    print(x.shape)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    obs = x
    act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    if normalize_data:

        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

if __name__ == '__main__':
    out = load_data("/home/khangtg/Documents/course/AI618_unsupervised_and_generative_models/code/ngsim_env/data/trajectories/ngsim.h5",
              )