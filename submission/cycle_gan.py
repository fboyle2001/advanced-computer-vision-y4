import time
import os
import shutil

import torch
import torch.nn as nn

from history_buffer import HistoryBuffer
from generator_model import Generator
from discriminator_model import PatchDiscriminator

"""
Implementation of the CycleGAN model

References:
Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
class CycleGAN:
    def __init__(
            self, 
            resnet_block_count, 
            upsample_strategy, 
            device,
            pool_size,
            opt_scheduler_type,
            max_epochs,
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

        # This from their repo
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

        if self.start_epoch == 0:
            print("Initialised weights")
            self._initialise_weights()

    # This is directly inspired by the CycleGAN repo, I make no claim to originality here at all
    def _initialise_weights(self):
        def applicator(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
        
        self.G.apply(applicator)
        self.F.apply(applicator)
        self.D_X.apply(applicator)
        self.D_Y.apply(applicator)
    
    def _step_learning_rate(self, name, scheduler, optimiser):
        previous_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimiser.param_groups[0]["lr"]
        print(f"Updated {name} learning rate from {previous_lr} to {new_lr}")

    def step_learning_rates(self):
        # Need to adjust learning rates according to scheduler
        self._step_learning_rate("G_opt", self.G_opt_scheduler, self.G_opt)
        self._step_learning_rate("F_opt", self.F_opt_scheduler, self.F_opt)
        self._step_learning_rate("D_X_opt", self.D_X_opt_scheduler, self.D_X_opt)
        self._step_learning_rate("D_Y_opt", self.D_Y_opt_scheduler, self.D_Y_opt)

    def apply(self, tensors, x_to_y):
        # Transfer style
        model = self.G if x_to_y else self.F
        model.eval()

        with torch.no_grad():
            processed_tensors = model(tensors.to(self.device).detach())
        
        model.train()
        return processed_tensors.detach().cpu()

    def save(self, epoch, full_save, folder=None):
        # Checkpointing
        folder_t = f"{self.save_folder}/{epoch if folder is None else folder}"

        if folder == "latest" and os.path.exists(folder_t) and os.path.isdir(folder_t):
            shutil.rmtree(folder_t)

        os.makedirs(folder_t, exist_ok=True)

        if full_save:
            torch.save({
                "epoch": epoch,
                "G": self.G.state_dict(),
                "F": self.F.state_dict(),
                "D_X": self.D_X.state_dict(),
                "D_Y": self.D_Y.state_dict(),
                "G_opt": self.G_opt.state_dict(),
                "F_opt": self.F_opt.state_dict(),
                "D_X_opt": self.D_X_opt.state_dict(),
                "D_Y_opt": self.D_Y_opt.state_dict(),
                "fake_X_buffer": self.fake_X_buffer.buffer,
                "fake_Y_buffer": self.fake_Y_buffer.buffer
            }, f"{folder_t}/checkpoint.tar")

        torch.save(self.G.state_dict(), f"{folder_t}/G.pth")
        torch.save(self.F.state_dict(), f"{folder_t}/F.pth")