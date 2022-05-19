import argparse
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

# from datasets import NGSIMLoader
from models import create_LatentODE_model
from trainer import Trainer
from config import get_cfg_defaults
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # setup GPU device if available, move model into configured device
    device, device_ids = prepare_device(config['n_gpus'])

    # setup data_loader instances
    # dataloader = NGSIMLoader(config["dataset"])
    # train_dataloader, test_dataloader = dataloader.split_train_test()

    # build model architecture, then print to console
    obsrv_std = torch.tensor([config.model.observ_std]).to(device)
    z0_prior = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.]).to(device))

    model = create_LatentODE_model(config.model, config.model.input_dim,
                                   z0_prior, obsrv_std, device)
    # self.model = model.to(self.device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=config["trainer"]["lr"], weight_decay=config["trainer"]["weight_decay"])

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, config["trainer"]["lr_step_size"],
                                             gamma=config["trainer"]["lr_decay"])

    trainer = Trainer(model, optimizer,
                      config=config, device=device,
                      # data_loader=train_dataloader,
                      # valid_data_loader=test_dataloader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-re', '--retrain', default=False, type=bool)
    args.add_argument('-p', '--pretrained', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = get_cfg_defaults()
    parse_args = args.parse_args()
    main(config)
    # if not parse_args.retrain:
    #     main(config)
    # else:
    #     retrain(config, parse_args.pretrained)
