import torch

from .base_trainer import BaseTrainer
from utils import to_device


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = 0
        self.len_epoch += len(data_loader.dataset)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 50 #int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        batch_idx, current = 0, 0
        for loader in self.data_loader:
            for data, target in loader:
                data, target = to_device(data, self.device), to_device(target, self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                if self.vgg16 is None:
                    loss = self.criterion(output, target)
                else:
                    loss = self.criterion(output, target, vgg=self.vgg16)
                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

                self.train_metrics.update('loss', loss.item(), n=loader.batch_size)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, target).item(), n=loader.batch_size)

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f} RMSE: {:.6f}'.format(
                        epoch,
                        self._progress(current),
                        self.train_metrics.avg('loss'),
                        self.train_metrics.avg(self.metric_ftns[1].__name__)
                    ))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
                batch_idx += 1
                current += loader.batch_size
        # for batch_idx, (data, target) in enumerate(self.data_loader):

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            batch_idx = 0
            for loader in self.valid_data_loader:
                for data, target in loader:
                    data, target = to_device(data, self.device), to_device(target, self.device)

                    output = self.model(data)
                    if self.vgg16 is None:
                        loss = self.criterion(output, target)
                    else:
                        loss = self.criterion(output, target, vgg=self.vgg16)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, target).item())
                    batch_idx += 1
            # for batch_idx, (data, target) in enumerate(self.valid_data_loader):

                # print(loss.item(), met(output, target).item())
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #   self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        # if hasattr(self.data_loader, 'n_samples'):
        #     current = batch_idx * self.data_loader.batch_size
        #     total = self.data_loader.n_samples
        # else:
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
