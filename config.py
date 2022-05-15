from yacs.config import CfgNode as CN

_C = CN()
_C.N_GPUS = 1
_C.LOG_DIR = "logs"
_C.SAVE_DIR = "logs/checkpoints"

_C.DATASET = CN()
_C.DATASET.DATA_PATH = ""
_C.DATASET.DATA_LIST = ""
_C.DATASET.ACT_KEYS = ""
_C.DATASET.BATCH_SIZE = 50
_C.DATASET.TEST_RATIO = 0.1
_C.DATASET.MIN_TRAJ_LENGTH = ""
_C.DATASET.NORMALIZE_DATA = True
_C.DATASET.ACT_LOW = -1.0
_C.DATASET.ACT_HIGH = 1.0
_C.DATASET.CLIP_STD_MULTIPLE = 1e2

_C.MODEL = CN()

_C.TRAINER = CN()
# Number of training epochs
_C.TRAINER.EPOCHS = 300
# Number of workers for doing things
_C.TRAINER.SAVE_PERIOD = 1
_C.TRAINER.RESUME = None
_C.TRAINER.LR = 0.01
_C.TRAINER.WEIGHT_DECAY = 0.0
_C.TRAINER.LR_STEP_SIZE = 50
_C.TRAINER.LR_DECAY = 0.5


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()