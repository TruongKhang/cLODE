import copy
import torch
from loguru import logger

from .base_trainer import BaseTrainer
from utils import to_device, MetricTracker
from datasets import NGSIMLoader


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, optimizer, config, device, lr_scheduler=None):
        super().__init__(model, optimizer, config, device)
        self.config = config
        self.data_files = config["list_dataset_files"]
        # self.data_loader = data_loader
        # self.len_epoch = len(data_loader)
        # self.valid_data_loader = valid_data_loader
        # self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('loss', 'rmse')
        self.valid_metrics = MetricTracker('loss', 'rmse')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.valid_metrics.reset()

        for file in self.data_files:
            cfg_dataset = copy.deepcopy(self.config["dataset"])
            dataloader = NGSIMLoader(cfg_dataset, file)
            train_loader, test_loader = dataloader.split_train_test()

            logger.info("Training dataset {}".format(file))
            for batch_idx, batch_dict in enumerate(train_loader):
                batch_dict = to_device(batch_dict, self.device)
                wait_until_kl_inc = 10
                if epoch < wait_until_kl_inc:
                    kl_coef = 0.
                else:
                    kl_coef = (1 - 0.99 ** (epoch - wait_until_kl_inc))

                self.optimizer.zero_grad()
                outputs = self.model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

                # self.writer.add_scalar("training loss", loss.item(), (epoch - 1) * self.len_epoch + batch_idx)
                # self.writer.add_scalar("training mse", outputs["mse"].item(), (epoch - 1) * self.len_epoch + batch_idx)

                self.train_metrics.update('loss', loss.item(), n=train_loader.batch_size)
                self.train_metrics.update("rmse", torch.sqrt(outputs["mse"]).item(), n=train_loader.batch_size)

                if batch_idx % self.log_step == 0:
                    logger.info('Train Epoch: {} {} Loss: {:.6f} RMSE: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx, len(train_loader)),
                        self.train_metrics.avg('loss'),
                        self.train_metrics.avg('rmse')
                    ))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # if batch_idx == self.len_epoch:
                #     break
                # batch_idx += 1
                # current += loader.batch_size

            # for batch_idx, (data, target) in enumerate(self.data_loader):

            # if self.do_validation:
            log = self.train_metrics.result()
            logger.info("Test on dataset {}".format(file))
            val_log = self._valid_epoch(epoch, test_loader)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            # print logged informations to the screen
            logger.info("training and test results for the dataset {}".format(file))
            for key, value in log.items():
                logger.info('    {:15s}: {}'.format(str(key), value))

        logger.info("The final training and test results for all datasets")
        log = self.train_metrics.result()
        val_log = self.valid_metrics.result()
        log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, valid_data_loader):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        # self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(valid_data_loader):
                batch_dict = to_device(batch_dict, self.device)

                outputs = self.model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=1.0)

                # self.writer.add_scalar("test loss", outputs["loss"].item(), (epoch - 1) * len(self.valid_data_loader) + batch_idx)
                # self.writer.add_scalar("test mse", outputs["mse"].item(), (epoch - 1) * len(self.valid_data_loader) + batch_idx)

                self.valid_metrics.update("loss", outputs["loss"].item())
                self.valid_metrics.update("rmse", torch.sqrt(outputs["mse"]).item())

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #   self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx, total):
        base = '[{}/{} ({:.0f}%)]'
        # if hasattr(self.data_loader, 'n_samples'):
        #     current = batch_idx * self.data_loader.batch_size
        #     total = self.data_loader.n_samples
        # else:
        current = batch_idx
        # total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
