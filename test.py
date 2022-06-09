import argparse
import numpy as np
import os
import torch
from torch.distributions.normal import Normal
import torch.multiprocessing as mp
import time
from loguru import logger


from utils import prepare_device, write_trajectories, to_device
from models import create_LatentODE_model
from datasets import NGSIMLoader, NGSIMDatasetSim
from config import get_cfg_defaults
from simulation.simulate import collect_trajectories
from visualization import Visualizations


parser = argparse.ArgumentParser(description='validation settings')
parser.add_argument('--test_mode', type=str, default='simulation', choices=["simulation", "visualization"])
parser.add_argument('--exp_dir', type=str, default='outputs')
parser.add_argument('--test_datapath', type=str, help="the h5 file includes the observation data")
parser.add_argument('--test_filename', type=str, default="trajdata_i101-22agents-0750am-0805am.txt")
parser.add_argument('--ngsim_filename', type=str, default="trajdata_i101_trajectories-0750am-0805am.txt")
parser.add_argument('--use_multi_agents', action="store_true", help="running simulation for multi-agent")
parser.add_argument('--n_envs', type=int, default=22, help="number of agents")
parser.add_argument('--ckpt_path', type=str, default="pretrained model to predict actions")
parser.add_argument('--max_obs_length', type=int, default=1000, help="number of simulations")
parser.add_argument('--n_procs', type=int, default=1, help="number of processes to run simulation in parallel")
parser.add_argument('--viz_obs_ratio', type=float, default=1.0, help="ratio of observed data for visualization")
parser.add_argument('--save_viz_figures', action="store_true", help="save visualizations")

args = parser.parse_args()


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_list(["dataset.test_data_path", args.test_datapath, "dataset.test_filename", args.test_filename,
                         "ngsim_env.ngsim_filename", args.ngsim_filename, "dataset.max_obs_length", args.max_obs_length,
                         "ngsim_env.n_envs", args.n_envs, "ngsim_env.env_multiagent", args.use_multi_agents])
    logger.info("config:{}".format(cfg))

    device, _ = prepare_device(1)

    if args.test_mode == 'simulation':
        torch.manual_seed(1995)
        mp.set_start_method("spawn", force=True)

    logger.info("Load pretrained model")
    obsrv_std = torch.tensor([0.01]).to(device)
    z0_prior = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.]).to(device))

    model = create_LatentODE_model(cfg.model, cfg.model.input_dim, z0_prior, obsrv_std, device)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.test_mode == 'simulation':
        logger.info("Initilize simulation dataloader with {} processes in parallel".format(args.n_procs))
        ngsim_loader = NGSIMDatasetSim(cfg.dataset, cfg.ngsim_env.ngsim_filename, use_multi_agents=args.use_multi_agents)
        obs_data = ngsim_loader.test_data  # shape [num_agents, num_observations, num_features]
        logger.info("shape of observation used for simulation: {}".format(obs_data.shape))

        start_time = time.time()
        if args.n_procs > 1:
            split_data_ids = np.array_split(np.arange(obs_data.shape[1] - 1), args.n_procs)
            model.share_memory()
            pool = mp.Pool(processes=args.n_procs)
            results = []
            for pid in range(args.n_procs):
                sub_obs_data = {"obs_data": obs_data[:, split_data_ids[pid]], "act_idxs": ngsim_loader.act_idxs,
                                "data_statistics": ngsim_loader.data_statistics}
                out = pool.apply_async(collect_trajectories, args=(cfg, model, sub_obs_data, device, pid))
                results.append(out)
            results = [out.get() for out in results]
            pool.close()
            time.sleep(10)
            results = np.concatenate(results, axis=0)
        else:
            data = {"obs_data": obs_data[:, :-1], "act_idxs": ngsim_loader.act_idxs,
                    "data_statistics": ngsim_loader.data_statistics}
            results = collect_trajectories(cfg, model, data, device)
        end_time = time.time()
        print("average of prediction time for each step of : ", (end_time - start_time) / (obs_data.shape[1] - 1))
        # results = pool.starmap(collect_trajectories, list_parallel_args)
        # pool.close()

        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        output_filepath = os.path.join(args.exp_dir, '{}_{}agents_cLatentODE.npz'.format(
            args.test_filename.split('.')[0],
            args.n_envs))
        write_trajectories(output_filepath, results)
    elif args.test_mode == 'visualization':
        logger.info("Initilize visualization dataloader")
        dataloader = NGSIMLoader(cfg.dataset, cfg.ngsim_env.ngsim_filename, mode='test')
        _, test_dataloader = dataloader.split_train_test(observed_ratio=args.viz_obs_ratio, test_batch_size=1)
        n_traj_to_show = 6
        list_batch_data = []
        for idx, batch_data in enumerate(test_dataloader):
            list_batch_data.append(to_device(batch_data, device))
            if (idx+1) == n_traj_to_show:
                break
        viz = Visualizations(device)
        viz.draw_all_plots_one_dim(list_batch_data, model, dim_to_show=0, save=args.save_viz_figures)
    else:
        raise "unknown test mode!"
