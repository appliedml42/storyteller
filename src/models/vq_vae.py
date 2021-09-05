import argparse
import logging
import math
import time
from abc import ABC
from argparse import Namespace
import horovod.torch as hvd
import torch
import wandb
from dalle_pytorch import DiscreteVAE
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


class VQ_VAE(ABC):
    def __init__(self, *args):
        if len(args) == 2:
            generic_args, other_args = args
            self.parser = argparse.ArgumentParser()
            if generic_args.which == 'train':
                self.parser.add_argument('--batch_size', type=int, required=True)
                self.parser.add_argument('--learning_rate', type=float, required=True)
                self.parser.add_argument('--lr_decay_rate', type=float, required=True)
                self.parser.add_argument('--num_tokens', type=int, required=True)
                self.parser.add_argument('--num_layers', type=int, required=True)
                self.parser.add_argument('--num_resnet_blocks', type=int, required=True)
                self.parser.add_argument('--smooth_l1_loss', default=False, action='store_true')
                self.parser.add_argument('--emb_dim', type=int, required=True)
                self.parser.add_argument('--hid_dim', type=int, required=True)
                self.parser.add_argument('--kl_loss_weight', type=float, required=True)
                self.parser.add_argument('--starting_temp', type=float, required=True)
                self.parser.add_argument('--temp_min', type=float, required=True)
                self.parser.add_argument('--anneal_rate', type=float, required=True)
                self.parser.add_argument('--num_workers', type=int, required=True)
                self.parser.add_argument('--prefetch_factor', type=int, required=True)
                self.parser.add_argument('--cache_duration', type=int, required=True)
                self.parser.add_argument('--log_tier1_interval', type=int, required=True)
                self.parser.add_argument('--log_tier2_interval', type=int, required=True)
                self.parser.add_argument('--save_interval', type=int, required=True)
                self.parser.add_argument('--num_images_save', type=int, required=True)

            self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
            self.initialize()
        elif isinstance(args[0], Namespace):
            self.params = args[0]
            self.initialize()

    def initialize(self):
        self.model = DiscreteVAE(image_size=self.params.image_size,
                                 num_layers=self.params.num_layers,
                                 num_tokens=self.params.num_tokens,
                                 codebook_dim=self.params.emb_dim,
                                 hidden_dim=self.params.hid_dim,
                                 num_resnet_blocks=self.params.num_resnet_blocks,
                                 smooth_l1_loss=self.params.smooth_l1_loss,
                                 kl_div_loss_weight=self.params.kl_loss_weight).cuda()

        if self.params.use_horovod:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        self.optimizer, self.schedule = self.get_optimizer()

    def train(self, dataset):
        sampler = None
        if self.params.use_horovod:
            sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        data_loader = DataLoader(dataset,
                                 batch_size=self.params.batch_size,
                                 sampler=sampler,
                                 num_workers=self.params.num_workers,
                                 prefetch_factor=self.params.prefetch_factor,
                                 shuffle=True)

        step = 0
        temp = self.params.starting_temp
        exp_config = vars(self.params)
        exp_config['num_params'] = sum(p.numel() for p in self.model.parameters())
        if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
            run = wandb.init(
                project='storyteller',
                name=self.params.experiment_dpath.split('/')[-1],
                config=exp_config
            )

        cached = False
        for epoch in range(self.params.epochs):
            if self.params.use_horovod:
                sampler.set_epoch(epoch)
            logging.info(f'Starting epoch {epoch}')
            for i, data in enumerate(data_loader):
                '''
                This starts the data loader and the caching process. Main training process sleeps for cache_duration 
                seconds. Thus, allowing the data loader processes to cache samples from the web.
                '''
                if not cached:
                    time.sleep(self.params.cache_duration)
                    cached = True
                images = data['image'].cuda()
                loss, recons = self.model(images, return_loss=True, return_recons=True, temp=temp)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % 100 == 0:
                    temp = max(temp * math.exp(-self.params.anneal_rate * step), self.params.temp_min)
                    self.schedule.step()
                if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                    logs = self.log_train(loss, recons, images, i, epoch)
                    wandb.log(logs)

                step += 1
            logging.info(f'Finished epoch {epoch}')
            cached = False

        if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
            wandb.save(f'{self.params.experiment_dpath}/*')

        wandb.finish()

    def get_optimizer(self):
        if self.params.use_horovod:
            optimizer = Adam(self.model.parameters(), lr=hvd.size() * self.params.learning_rate)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=self.model.named_parameters(),
                                                 compression=hvd.Compression.fp16,
                                                 op=hvd.Average,
                                                 gradient_predivide_factor=1.0)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.params.learning_rate)

        schedule = ExponentialLR(optimizer=optimizer, gamma=self.params.lr_decay_rate)
        return optimizer, schedule

    def log_train(self, loss, recons, train_images, step, epoch):
        logs = {}
        if step % self.params.log_tier2_interval == 0:
            with torch.no_grad():
                codes = self.model.get_codebook_indices(train_images[:self.params.num_images_save])
                hard_recons = self.model.decode(codes)
            images, recons = train_images[:self.params.num_images_save], recons[:self.params.num_images_save]
            images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
            images, recons, hard_recons = map(
                lambda t: make_grid(t.float(), nrow=int(math.sqrt(self.params.num_images_save)), normalize=True,
                                    range=(-1, 1)),
                (images, recons, hard_recons))

            logs = {
                **logs,
                'sample images': wandb.Image(images, caption='original images'),
                'reconstructions': wandb.Image(recons, caption='reconstructions'),
                'hard reconstructions': wandb.Image(hard_recons, caption='hard reconstructions'),
                'codebook_indices': wandb.Histogram(codes)
            }

        if step % self.params.log_tier1_interval == 0:
            lr = self.schedule.get_last_lr()[0]
            logging.info(f'Epoch:{epoch} Step:{step} loss:{loss.item()} lr:{lr}')
            logs = {
                **logs,
                'epoch': epoch,
                'step': step,
                'loss': loss.item(),
                'lr': lr
            }

        if step % self.params.save_interval == 0:
            if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                save_obj = {
                    'weights': self.model.state_dict()
                }
                torch.save(save_obj, f'{self.params.experiment_dpath}/vae_{step}.pt')

        return logs
