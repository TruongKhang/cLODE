import os.path

import torch
import torch.nn as nn
from abc import abstractmethod
from tensorboardX import SummaryWriter
from loguru import logger


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, optimizer, config, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = device

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config["save_dir"]
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # setup visualization writer instance
        if not os.path.exists(config["log_dir"]):
            os.makedirs(config["log_dir"])
        self.writer = SummaryWriter(config["log_dir"])

        if cfg_trainer["resume"] is not None:
            self._resume_checkpoint(cfg_trainer["resume"])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                logger.info('    {:15s}: {}'.format(str(key), value))

            # save model
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = '{}/checkpoint-epoch{}.ckpt'.format(self.checkpoint_dir, epoch)
        torch.save(state, filename)
        logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            logger.warning("Warning: Architecture configuration given in config file is different from that of "
                           "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                           "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))