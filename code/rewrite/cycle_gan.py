from typing import List

import time
import os

import torch
import torch.nn as nn

from utils import HistoryBuffer
from generator_model import Generator
from discriminator_model import PatchDiscriminator

class CycleGAN:
    def __init__(
            self, 
            resnet_block_count, 
            upsample_strategy, 
            device,
            pool_size,
            opt_scheduler_type,
            max_epochs,
            save_frequency=5,
            start_epoch=0, 
            save_folder=None
        ):

        self.device = device
        self.resnet_block_count = resnet_block_count

        # Networks

        self.G = Generator(resnet_blocks=resnet_block_count, up_type=upsample_strategy).to(device)
        self.F = Generator(resnet_blocks=resnet_block_count, up_type=upsample_strategy).to(device)

        self.D_X = PatchDiscriminator().to(device)
        self.D_Y = PatchDiscriminator().to(device)

        # Buffers

        self.fake_X_buffer = HistoryBuffer(pool_size)
        self.fake_Y_buffer = HistoryBuffer(pool_size)

        # Losses

        self.gan_loss = nn.MSELoss().to(device)
        self.cycle_loss = nn.L1Loss().to(device)
        self.identity_loss = nn.L1Loss().to(device)

        # Optimisers

        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.F_opt = torch.optim.Adam(self.F.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.D_X_opt = torch.optim.Adam(self.D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_Y_opt = torch.optim.Adam(self.D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Learning Rate Schedulers

        lambda_rule = lambda epoch: 1

        # TODO: CITE
        if opt_scheduler_type == "linear_decay_with_warmup":
            lambda_rule = lambda epoch: 1.0 - max(0, epoch - max_epochs // 2 - 1) / float(max_epochs // 2 + 1)
        
        self.G_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_opt, lr_lambda=lambda_rule)
        self.F_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.F_opt, lr_lambda=lambda_rule)

        self.D_X_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_X_opt, lr_lambda=lambda_rule)
        self.D_Y_opt_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_Y_opt, lr_lambda=lambda_rule)

        # Save Folder
        if save_folder is not None:
            self.save_folder = save_folder
        else:
            self.save_folder = f"./runs/CycleGAN/{time.time()}"
            os.makedirs(self.save_folder)
        
        self.start_epoch = start_epoch
    
    def _step_learning_rate(self, name, scheduler, optimiser):
        previous_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimiser.param_groups[0]["lr"]
        print(f"Updated {name} learning rate from {previous_lr} to {new_lr}")

    def step_learning_rates(self):
        self._step_learning_rate("G_opt", self.G_opt_scheduler, self.G_opt)
        self._step_learning_rate("F_opt", self.F_opt_scheduler, self.F_opt)
        self._step_learning_rate("D_X_opt", self.D_X_opt_scheduler, self.D_X_opt)
        self._step_learning_rate("D_Y_opt", self.D_Y_opt_scheduler, self.D_Y_opt)