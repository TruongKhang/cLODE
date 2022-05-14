from yacs.config import CfgNode as CN

_C = CN()
_C.N_GPUS = 1
_C.LOG_DIR = "logs"
_C.SAVE_DIR = "logs/checkpoints"

_C.TRAINER = CN()
# Number of training epochs
_C.TRAINER.EPOCHS = 8
# Number of workers for doing things
_C.TRAINER.SAVE_PERIOD = 1
_C.TRAINER.RESUME = None

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.HYPERPARAMETER_1 = 0.1
# The all important scales for the stuff
_C.TRAIN.SCALES = (2, 4, 8, 16)


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()