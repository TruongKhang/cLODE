import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import torch
from torch.distributions.normal import Normal
import time
from loguru import logger

backend = 'TkAgg'
import matplotlib

matplotlib.use(backend)
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import hgail.misc.simulation
import hgail.misc.utils

from utils import to_device, prepare_device, write_trajectories
from simulator.env import build_ngsim_env
from models import create_LatentODE_model
from datasets import NGSIMLoader
from config import get_cfg_defaults

plt.style.use("ggplot")


parser = argparse.ArgumentParser(description='validation settings')
parser.add_argument('--exp_dir', type=str, default='outputs')
parser.add_argument('--test_datapath', type=str, help="the h5 file includes the observation data")
parser.add_argument('--test_filename', type=str, default="trajdata_i101-22agents-0750am-0805am.txt")
parser.add_argument('--ngsim_filename', type=str, default="trajdata_i101_trajectories-0750am-0805am.txt")
parser.add_argument('--use_multiagent', action="store_true", help="running simulation for multi-agent")
parser.add_argument('--n_envs', type=int, default=22, help="number of agents")
parser.add_argument('--ckpt_path', type=str, default="pretrained model to predict actions")
parser.add_argument('--max_obs_length', type=int, default=1000, help="number of simulations")

args = parser.parse_args()


def online_predict(env, model, dataloader, device, action_idxs):
    env_kwargs = {}
    _ = env.reset(**env_kwargs)
    predicted_trajs = []

    avg = 0
    with torch.no_grad():
        for step, data_dict in enumerate(dataloader):
            if step % 10 == 0:
                print(step)
            start = time.time()

            data_dict = to_device(data_dict, device)
            # data = data_dict["data_to_predict"]
            # time_steps = data_dict["tp_to_predict"]
            # mask = data_dict["mask_predicted_data"]

            observed_data = data_dict["observed_data"]
            # observed_time_steps = data_dict["observed_tp"]
            # observed_mask = data_dict["observed_mask"]
            # time_steps_to_predict = data_dict["tp_to_predict"]
            time_steps = data_dict["time_steps"]
            # observed_time_steps = time_steps[[0]]
            # time_steps_to_predict = time_steps[[1]]
            # observed_mask = torch.ones_like(observed_data)

            traj = hgail.misc.simulation.Trajectory()
            start_obs_ts = 0
            for end_obs_ts in range(1, len(time_steps)):
                if end_obs_ts > 10:
                    start_obs_ts = end_obs_ts - 10
                observed_time_steps = time_steps[start_obs_ts:end_obs_ts]
                time_steps_to_predict = time_steps[end_obs_ts:(end_obs_ts + 1)]
                observed_data_used = observed_data[:, start_obs_ts:end_obs_ts]
                observed_mask = torch.ones_like(observed_data_used)
                pred_actions, info = model.get_reconstruction(time_steps_to_predict, observed_data_used,
                                                              observed_time_steps, mask=observed_mask,
                                                              n_traj_samples=1)
                mean_actions, std_actions = pred_actions.mean(dim=0), pred_actions.std(dim=0)
                mean_actions, std_actions = mean_actions.cpu().numpy(), std_actions.cpu().numpy()

                mean_act, std_act = mean_actions[:, 0], std_actions[:, 0]
                agent_info = {"mean": mean_act, "log_std": np.log(std_act)}

                nx, r, dones, env_info = env.step(mean_act)
                nx[:, action_idxs] = np.clip(nx[:, action_idxs], -1, 1) # normalize observed actions in range [-1, 1]
                traj.add(observed_data[:, -1].cpu().numpy(), mean_act, r, agent_info, env_info)
                if any(dones): break

                cur_obs = torch.from_numpy(nx).float().to(device)
                observed_data = torch.cat((observed_data, cur_obs.unsqueeze(1)), dim=1)
                # observed_mask = torch.ones_like(observed_data)

            _ = env.reset(**env_kwargs)
            traj = traj.flatten()
            # if step == 0:
            #    print(np.mean(traj["rmse_pos"], axis=1))
            #    exit(0)
            predicted_trajs.append(traj)

            end = time.time()
            avg += (end - start)

    print("average of prediction time for each step: ", avg / len(dataloader))

    return predicted_trajs


def collect_trajectories(config):
    trajlist = []

    ngsim_env_args = config.ngsim_env
    logger.info("Build ngsim environment for simulation")
    env, _, _ = build_ngsim_env(ngsim_env_args, alpha=0.)

    device, _ = prepare_device(1)

    logger.info("Load pretrained model")
    obsrv_std = torch.tensor([0.01]).to(device)
    z0_prior = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.]).to(device))

    model = create_LatentODE_model(config.model, config.model.input_dim,
                                   z0_prior, obsrv_std, device)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    logger.info("Initilize dataloader")
    dataloader = NGSIMLoader(config.dataset, config.ngsim_env.ngsim_filename, mode="test").get_test_dataloader()
    action_idxs = dataloader.dataset.act_idxs

    # normalizing observation data when running simulation
    normalized_env = hgail.misc.utils.extract_normalizing_env(env)
    if normalized_env is not None:
        obs_mean, obs_std = dataloader.dataset.data_statistics
        print("mean observation shape: {}, std observation shape: {}".format(obs_mean.shape, obs_std.shape))
        normalized_env._obs_mean = obs_mean
        normalized_env._obs_var = obs_std ** 2

    logger.info("Predict trajectories")
    traj = online_predict(env, model, dataloader, device, action_idxs)
    trajlist.append(traj)

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    output_filepath = os.path.join(args.exp_dir, '{}_1trajs_{}agents_10observed_time_steps_ode_traj.npz'.format(args.test_filename.split('.')[0],
                                                                                   args.n_envs))
    write_trajectories(output_filepath, trajlist)

    return trajlist


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_list(["dataset.test_data_path", args.test_datapath, "dataset.test_filename", args.test_filename,
                         "ngsim_env.ngsim_filename", args.ngsim_filename, "dataset.max_obs_length", args.max_obs_length,
                         "ngsim_env.n_envs", args.n_envs, "ngsim_env.env_multiagent", args.use_multiagent])
    logger.info("config:{}".format(cfg))
    collect_trajectories(cfg)
