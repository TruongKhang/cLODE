import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import torch
from torch.distributions.normal import Normal
import time

backend = 'TkAgg'
import matplotlib

matplotlib.use(backend)
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from contexttimer import Timer

import hgail.misc.simulation
import hgail.misc.utils

from utils import to_device, prepare_device
from models.modules.utils import linspace_vector
from .env import build_ngsim_env
from models import create_LatentODE_model
from datasets import NGSIMLoader

plt.style.use("ggplot")


def online_predict(env, model, dataloader, config):
    env_kwargs = {}
    _ = env.reset(**env_kwargs)
    predicted_trajs = []

    avg = 0
    for step, data_dict in enumerate(dataloader):
        if step % 100 == 0:
            print(step)
        start = time.time()

        data_dict = to_device(data_dict)
        # data = data_dict["data_to_predict"]
        time_steps = data_dict["tp_to_predict"]
        # mask = data_dict["mask_predicted_data"]

        observed_data = data_dict["observed_data"]
        observed_time_steps = data_dict["observed_tp"]
        observed_mask = data_dict["observed_mask"]

        time_steps_to_predict = linspace_vector(time_steps[0], time_steps[-1], config["env_H"])

        pred_actions, info = model.get_reconstruction(time_steps_to_predict,
                                                      observed_data, observed_time_steps, mask=observed_mask,
                                                      n_traj_samples=10)

        traj = ngsim_estimate(pred_actions, observed_data[:, -1], env, env_kwargs)
        predicted_trajs.append(traj)

        end = time.time()
        avg += (start - end)

    # print(avg / (max_steps - 1))
    # for i in range(n_agents):
    #     plt.plot(range(step + 1), d[i, :])
    # plt.show()

    return predicted_trajs


def ngsim_estimate(pred_actions, obs, env, env_kwargs):
    traj = hgail.misc.simulation.Trajectory()

    mean_actions, std_actions = pred_actions.mean(dim=0), pred_actions.std(dim=0)
    mean_actions, std_actions = mean_actions.cpu().numpy(), std_actions.cpu().numpy()
    n_time_steps = mean_actions.shape[1]
    for i in range(n_time_steps):
        mean_act, std_act = mean_actions[:, i], std_actions[:, i]
        agent_info = {"mean": mean_act, "log_std": np.log(std_act)}

        nx, r, dones, env_info = env.step(mean_act)
        traj.add(obs, mean_act, r, agent_info, env_info)
        if any(dones): break
        obs = nx

    # this should be delete and replaced
    _ = env.reset(**env_kwargs)

    return traj.flatten()


def collect_trajectories(config, egoids, starts, trajlist, pid):
    ngsim_env_args = config.ngsim_env
    env, _, _ = build_ngsim_env(ngsim_env_args, alpha=0.)

    device, _ = prepare_device(1)

    obsrv_std = torch.tensor([0.01]).to(device)
    z0_prior = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.]).to(device))

    model = create_LatentODE_model(config.model, config.model.input_dim,
                                   z0_prior, obsrv_std, device)
    ckpt = torch.load(config["ckpt_path"])
    model.load_state_dict(ckpt["state_dict"])

    dataloader = NGSIMLoader(config.dataset, config.dataset.file_test).get_test_dataloader()

    traj = online_predict(env, model, dataloader, config)
    trajlist.append(traj)

    # collect trajectories
    # nids = len(egoids)
    #
    # if args.env_multiagent:
    #
    #     data = validate_utils.get_multiagent_ground_truth()
    # else:
    #     data = validate_utils.get_ground_truth()
    #     sample = np.random.choice(data['observations'].shape[0], 2)
    #
    # kwargs = dict()
    # if args.env_multiagent:
    #     traj = online_predict(
    #         env,
    #         policy,
    #         max_steps=max_steps,
    #         obs=data['observations'],
    #         mean=data['actions'],
    #         env_kwargs=kwargs,
    #         lbd=lbd,
    #         adapt_steps=adapt_steps
    #     )
    #     trajlist.append(traj)
    # else:
    #     for i in sample:
    #         sys.stdout.write('\rpid: {} traj: {} / {}'.format(pid, i, nids))
    #
    #         traj = online_adaption(
    #             env,
    #             policy,
    #             max_steps=max_steps,
    #             obs=data['observations'][i, :, :],
    #             mean=data['actions'][i, :, :],
    #             env_kwargs=kwargs,
    #             lbd=lbd,
    #             adapt_steps=adapt_steps
    #         )
    #         trajlist.append(traj)

    return trajlist


def parallel_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc,
        env_fn=utils.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None,
        lbd=0.99,
        adapt_steps=1):
    # build manager and dictionary mapping ego ids to list of trajectories
    manager = mp.Manager()
    trajlist = manager.list()

    # set policy function
    policy_fn = utils.build_hierarchy if use_hgail else validate_utils.build_policy

    # partition egoids
    proc_egoids = utils.partition_list(egoids, n_proc)

    # pool of processes, each with a set of ego ids
    pool = mp.Pool(processes=n_proc)

    # run collection
    results = []
    for pid in range(n_proc):
        res = pool.apply_async(
            collect_trajectories,
            args=(
                args,
                params,
                proc_egoids[pid],
                starts,
                trajlist,
                pid,
                env_fn,
                policy_fn,
                max_steps,
                use_hgail,
                random_seed,
                lbd,
                adapt_steps
            )
        )
        results.append(res)

    # wait for the processes to finish
    [res.get() for res in results]
    pool.close()
    # let the julia processes finish up
    time.sleep(10)
    return trajlist


def single_process_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc,
        env_fn=utils.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None):
    '''
    This function for debugging purposes
    '''
    # build list to be appended to
    trajlist = []

    # set policy function
    policy_fn = utils.build_hierarchy if use_hgail else validate_utils.build_policy
    tf.reset_default_graph()

    # collect trajectories in a single process
    collect_trajectories(
        args,
        params,
        egoids,
        starts,
        trajlist,
        n_proc,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed
    )
    return trajlist


def collect(
        egoids,
        starts,
        args,
        exp_dir,
        use_hgail,
        params_filename,
        n_proc,
        max_steps=200,
        collect_fn=parallel_collect_trajectories,
        random_seed=None,
        lbd=0.99,
        adapt_steps=1):
    '''
    Description:
        - prepare for running collection in parallel
        - multiagent note: egoids and starts are not currently used when running
            this with args.env_multiagent == True
    '''
    # load information relevant to the experiment
    params_filepath = os.path.join(exp_dir, 'imitate/log/{}'.format(params_filename))
    params = hgail.misc.utils.load_params(params_filepath)
    # validation setup
    validation_dir = os.path.join(exp_dir, 'imitate', 'validation')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_AGen.npz'.format(
        args.ngsim_filename.split('.')[0]))

    with Timer():
        trajs = collect_fn(
            args,
            params,
            egoids,
            starts,
            n_proc,
            max_steps=max_steps,
            use_hgail=use_hgail,
            random_seed=random_seed,
            lbd=0.99,
            adapt_steps=1
        )

    utils.write_trajectories(output_filepath, trajs)


def load_egoids(filename, args, n_runs_per_ego_id=10, env_fn=utils.build_ngsim_env):
    offset = args.env_H + args.env_primesteps
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data/')
    ids_filename = filename.replace('.txt', '-index-{}-ids.h5'.format(offset))
    ids_filepath = os.path.join(basedir, ids_filename)
    if not os.path.exists(ids_filepath):
        # this should create the ids file
        env_fn(args)
        if not os.path.exists(ids_filepath):
            raise ValueError('file unable to be created, check args')
    ids = np.array(h5py.File(ids_filepath, 'r')['ids'].value)

    # we want to sample start times uniformly from the range of possible values
    # but we also want these start times to be identical for every model we
    # validate. So we sample the start times a single time, and save them.
    # if they exist, we load them in and reuse them
    start_times_filename = filename.replace('.txt', '-index-{}-starts.h5'.format(offset))
    start_times_filepath = os.path.join(basedir, start_times_filename)
    # check if start time filepath exists
    if os.path.exists(start_times_filepath):
        # load them in
        starts = np.array(h5py.File(start_times_filepath, 'r')['starts'].value)
    # otherwise, sample the start times and save them
    else:
        ids_file = h5py.File(ids_filepath, 'r')
        ts = ids_file['ts'].value
        # subtract offset gives valid end points
        te = ids_file['te'].value - offset
        starts = np.array([np.random.randint(s, e + 1) for (s, e) in zip(ts, te)])
        # write to file
        starts_file = h5py.File(start_times_filepath, 'w')
        starts_file.create_dataset('starts', data=starts)
        starts_file.close()

    # create a dict from id to start time
    id2starts = dict()
    for (egoid, start) in zip(ids, starts):
        id2starts[egoid] = start

    ids = np.tile(ids, n_runs_per_ego_id)
    return ids, id2starts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, default='../../data/experiments/gail/')
    parser.add_argument('--params_filename', type=str, default='itr_2000.npz')
    parser.add_argument('--n_runs_per_ego_id', type=int, default=10)
    parser.add_argument('--use_hgail', type=str2bool, default=False)
    parser.add_argument('--use_multiagent', type=str2bool, default=False)
    parser.add_argument('--n_multiagent_trajs', type=int, default=10000)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--remove_ngsim_vehicles', type=str2bool, default=False)
    parser.add_argument('--lbd', type=float, default=0.99)
    parser.add_argument('--adapt_steps', type=int, default=1)

    run_args = parser.parse_args()

    args_filepath = os.path.join(run_args.exp_dir, 'imitate/log/args.npz')
    args = hyperparams.load_args(args_filepath)

    if run_args.use_multiagent:
        args.env_multiagent = True
        args.remove_ngsim_vehicles = run_args.remove_ngsim_vehicles

    if run_args.debug:
        collect_fn = single_process_collect_trajectories
    else:
        collect_fn = parallel_collect_trajectories

    filenames = [
        "trajdata_i101_trajectories-0750am-0805am.txt"
    ]

    if run_args.n_envs:
        args.n_envs = run_args.n_envs
    # args.env_H should be 200
    sys.stdout.write('{} vehicles with H = {}'.format(args.n_envs, args.env_H))

    for fn in filenames:
        args.ngsim_filename = fn
        if args.env_multiagent:
            # args.n_envs gives the number of simultaneous vehicles
            # so run_args.n_multiagent_trajs / args.n_envs gives the number
            # of simulations to run overall
            egoids = list(range(int(run_args.n_multiagent_trajs / args.n_envs)))
            starts = dict()
        else:
            egoids, starts = load_egoids(fn, args, run_args.n_runs_per_ego_id)
        collect(
            egoids,
            starts,
            args,
            exp_dir=run_args.exp_dir,
            max_steps=200,
            params_filename=run_args.params_filename,
            use_hgail=run_args.use_hgail,
            n_proc=run_args.n_proc,
            collect_fn=collect_fn,
            random_seed=run_args.random_seed,
            lbd=run_args.lbd,
            adapt_steps=run_args.adapt_steps
        )