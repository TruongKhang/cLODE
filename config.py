from yacs.config import CfgNode as CN

_C = CN()
_C.n_gpus = 1
_C.log_dir = "logs"
_C.save_dir = "logs/checkpoints"

_C.dataset = CN()
_C.dataset.data_path = "datasets/ngsim.h5"
_C.dataset.list_dataset_files = ["trajdata_i101_trajectories-0750am-0805am.txt"] #,
                                 #"trajdata_i101_trajectories-0805am-0820am.txt",
                                 #"trajdata_i101_trajectories-0820am-0835am.txt",
                                 #"trajdata_i80_trajectories-0400-0415.txt",
                                 #"trajdata_i80_trajectories-0500-0515.txt",
                                 #"trajdata_i80_trajectories-0515-0530.txt"]
_C.dataset.act_keys = ['accel', 'turn_rate_global']
_C.dataset.batch_size = 50
_C.dataset.test_ratio = 0.1
_C.dataset.min_traj_length = 250
_C.dataset.normalize_data = True
_C.dataset.act_low = [-4.0, -0.15]
_C.dataset.act_high = [4.0, 0.15]
_C.dataset.clip_std_multiple = 10

_C.model = CN()
_C.model.input_dim = 66
_C.model.output_dim = 2
_C.model.latents = 16
_C.model.rec_dims = 100
_C.model.rec_layers = 4
_C.model.gen_layers = 2
_C.model.units = 256
_C.model.gru_units = 128
_C.model.z0_encoder = "odernn"
_C.model.observ_std = 0.01

_C.trainer = CN()
# Number of training epochs
_C.trainer.epochs = 300
# Number of workers for doing things
_C.trainer.save_period = 1
_C.trainer.log_step = 5
_C.trainer.resume = None
_C.trainer.lr = 0.001
_C.trainer.weight_decay = 0.01
_C.trainer.lr_step_size = 50
_C.trainer.lr_decay = 0.5

_C.ngsim_env = CN()
_C.ngsim_env.ngsim_data_dir = "~/.julia/packages/NGSIM/9OYUa/data"
_C.ngsim_env.ngsim_filename = "trajdata_i101_trajectories-0750am-0805am.txt"
_C.ngsim_env.env_H = 200
_C.ngsim_env.env_primesteps = 50
_C.ngsim_env.env_action_repeat = 1
_C.ngsim_env.n_envs = 22 # number of agents
_C.ngsim_env.remove_ngsim_veh = False
_C.ngsim_env.env_reward = 0
_C.ngsim_env.env_multiagent = True



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
