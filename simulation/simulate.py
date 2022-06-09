import numpy as np
import torch
import time
from loguru import logger

import hgail.misc.simulation
import hgail.misc.utils

from simulation.env import build_ngsim_env


def online_predict(env, model, obs_data, device, action_idxs, num_ts=200, norm_ts=1000):
    """
    :param env:
    :param model:
    :param obs_data: use each sample in this data to start a simulation. total of obsverations = total of simulations
    :param device:
    :param action_idxs:
    :param norm_ts:
    :param num_ts:
    :return:
    """
    env_kwargs = {}
    _ = env.reset(**env_kwargs)
    # predicted_trajs = []
    trajlist = []

    obs_data = torch.from_numpy(obs_data).float().to(device)
    time_steps = torch.arange(num_ts + 1, dtype=torch.float32, device=device) / norm_ts

    avg = 0
    with torch.no_grad():
        for obs_idx in range(obs_data.shape[1]):
            if obs_idx % 10 == 0:
                print(obs_idx)
            start = time.time()
            obs_sample = obs_data[:, obs_idx]
            # data_dict = {"observed_data": torch.from_numpy(obs_sample).unsqueeze(1).float(),
            #              "time_steps": torch.from_numpy(time_steps).float()}

            # data_dict = to_device(data_dict, device)
            # data = data_dict["data_to_predict"]
            # time_steps = data_dict["tp_to_predict"]
            # mask = data_dict["mask_predicted_data"]

            observed_data = obs_sample.unsqueeze(1) #torch.from_numpy(obs_sample).unsqueeze(1).float().to(device) #data_dict["observed_data"]
            # observed_time_steps = data_dict["observed_tp"]
            # observed_mask = data_dict["observed_mask"]
            # time_steps_to_predict = data_dict["tp_to_predict"]
            # time_steps = torch.from_numpy(time_steps).float().to(device) #data_dict["time_steps"]
            # observed_time_steps = time_steps[[0]]
            # time_steps_to_predict = time_steps[[1]]
            # observed_mask = torch.ones_like(observed_data)

            traj = hgail.misc.simulation.Trajectory()
            start_obs_ts = 0
            for end_obs_ts in range(1, time_steps.shape[0]):
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
                nx[:, action_idxs] = np.clip(nx[:, action_idxs], -1, 1)  # normalize observed actions in range [-1, 1]
                traj.add(observed_data[:, -1].cpu().numpy(), mean_act, r, agent_info, env_info)
                if any(dones):
                    break

                cur_obs = torch.from_numpy(nx).float().to(device)
                observed_data = torch.cat((observed_data, cur_obs.unsqueeze(1)), dim=1)
                # observed_mask = torch.ones_like(observed_data)

            _ = env.reset(**env_kwargs)
            traj = traj.flatten()
            # if obs_idx == 0:
            #    print(np.mean(traj["rmse_pos"], axis=1))
            #    exit(0)
            trajlist.append(traj)

            end = time.time()
            avg += (end - start)

    return trajlist


def collect_trajectories(config, model, data, device, pid=0):

    ngsim_env_args = config.ngsim_env
    logger.info("Build ngsim environment for simulation, pid = {}".format(pid))
    env, _, _ = build_ngsim_env(ngsim_env_args, alpha=0.)

    action_idxs = data["act_idxs"] #sim_dataloader.dataset.act_idxs

    # normalizing observation data when running simulation
    normalized_env = hgail.misc.utils.extract_normalizing_env(env)
    if normalized_env is not None:
        obs_mean, obs_std = data["data_statistics"] #sim_dataloader.dataset.data_statistics
        print("mean observation shape: {}, std observation shape: {}".format(obs_mean.shape, obs_std.shape))
        normalized_env._obs_mean = obs_mean
        normalized_env._obs_var = obs_std ** 2

    # sim_dataloader = sim_dataloader.get_test_dataloader(pid)
    logger.info("Predict trajectories")
    trajlist = online_predict(env, model, data["obs_data"], device, action_idxs)
    # trajlist.append(traj)

    return trajlist
