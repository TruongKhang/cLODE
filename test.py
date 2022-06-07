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


from utils import prepare_device, write_trajectories, to_device
from models import create_LatentODE_model
from datasets import NGSIMLoader
from config import get_cfg_defaults
from simulation.simulate import collect_trajectories
from visualization import Visualizations


parser = argparse.ArgumentParser(description='validation settings')
parser.add_argument('--test_mode', type=str, default='simulation', choices=["simulation", "visualization"])
parser.add_argument('--exp_dir', type=str, default='outputs')
parser.add_argument('--test_datapath', type=str, help="the h5 file includes the observation data")
parser.add_argument('--test_filename', type=str, default="trajdata_i101-22agents-0750am-0805am.txt")
parser.add_argument('--ngsim_filename', type=str, default="trajdata_i101_trajectories-0750am-0805am.txt")
parser.add_argument('--use_multiagent', action="store_true", help="running simulation for multi-agent")
parser.add_argument('--n_envs', type=int, default=22, help="number of agents")
parser.add_argument('--ckpt_path', type=str, default="pretrained model to predict actions")
parser.add_argument('--max_obs_length', type=int, default=1000, help="number of simulations")
parser.add_argument('--n_procs', type=int, default=1, help="number of processes to run simulation in parallel")
parser.add_argument('--viz_obs_ratio', type=float, default=0.75, help="ratio of observed data for visualization")

args = parser.parse_args()


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_list(["dataset.test_data_path", args.test_datapath, "dataset.test_filename", args.test_filename,
                         "ngsim_env.ngsim_filename", args.ngsim_filename, "dataset.max_obs_length", args.max_obs_length,
                         "ngsim_env.n_envs", args.n_envs, "ngsim_env.env_multiagent", args.use_multiagent])
    logger.info("config:{}".format(cfg))

    device, _ = prepare_device(1)

    logger.info("Load pretrained model")
    obsrv_std = torch.tensor([0.01]).to(device)
    z0_prior = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.]).to(device))

    model = create_LatentODE_model(cfg.model, cfg.model.input_dim, z0_prior, obsrv_std, device)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.test_mode == 'simulation':
        logger.info("Initilize simulation dataloader with {} processes in parallel".format(args.n_procs))
        list_dataloader = NGSIMLoader(cfg.dataset, cfg.ngsim_env.ngsim_filename,
                                      mode="test").get_test_dataloader(args.n_procs)
        pool = mp.Pool(processes=args.n_procs)
        trajlist = mp.Manager().list()
        results = []
        for pid in range(args.n_procs):
            out = pool.apply_async(collect_trajectories, args=(cfg, model, list_dataloader[pid], device, trajlist))
            results.append(out)
        results = [out.get() for out in results]
        pool.close()
        time.sleep(10)

        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        output_filepath = os.path.join(args.exp_dir, '{}_{}agents_cLatentODE.npz'.format(
            args.test_filename.split('.')[0],
            args.n_envs))
        write_trajectories(output_filepath, trajlist)
    elif args.test_mode == 'visualization':
        logger.info("Initilize visualization dataloader")
        dataloader = NGSIMLoader(cfg.dataset, cfg.ngsim_env.ngsim_filename)
        _, test_dataloader = dataloader.split_train_test(observed_ratio=args.viz_obs_ratio, test_batch_size=5)
        batch_data = next(iter(test_dataloader))
        viz = Visualizations(device)
        viz.draw_all_plots_one_dim(to_device(batch_data, device), model, dim_to_show=0)
    else:
        raise "unknown test mode!"
