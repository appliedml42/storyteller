import argparse
import json
import logging
import math
from abc import ABC
from argparse import Namespace

import horovod.torch as hvd
import torch
import wandb
from dalle_pytorch import DALLE as DALLE_MODEL
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from common.utils import get_model_class


class DALLE(ABC):
    def __init__(self, generic_args, other_args):
        self.parser = argparse.ArgumentParser()
        if generic_args.which == 'train':
            self.parser.add_argument('--vae_config_fpath', type=str, required=True)
            self.parser.add_argument('--vae_weights_fpath', type=str, required=True)
            self.parser.add_argument('--batch_size', type=int, required=True)
            self.parser.add_argument('--learning_rate', type=float, required=True)
            self.parser.add_argument('--num_text_tokens', type=int, required=True)
            self.parser.add_argument('--text_seq_len', type=int, required=True)
            self.parser.add_argument('--dim', type=int, required=True)
            self.parser.add_argument('--depth', type=int, required=True)
            self.parser.add_argument('--heads', type=int, required=True)
            self.parser.add_argument('--dim_head', type=int, required=True)
            self.parser.add_argument('--reversible', default=False, action='store_true')
            self.parser.add_argument('--num_workers', type=int, required=True)
            self.parser.add_argument('--log_tier1_interval', type=int, required=True)
            self.parser.add_argument('--log_tier2_interval', type=int, required=True)
            self.parser.add_argument('--save_interval', type=int, required=True)

        self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)

        with open(self.params.vae_config_fpath) as reader:
            experiment_configuration = Namespace(**json.load(reader))
            model_class = get_model_class(experiment_configuration, False)
            experiment_configuration.use_horovod = False
            model = model_class(experiment_configuration)
            self.vae = model.model
            weights = torch.load(self.params.vae_weights_fpath)['weights']
            self.vae.load_state_dict(weights)

        self.model = DALLE_MODEL(
            vae=self.vae,
            num_text_tokens=self.params.num_text_tokens,
            text_seq_len=self.params.text_seq_len,
            dim=self.params.dim,
            depth=self.params.depth,
            heads=self.params.heads,
            dim_head=self.params.dim_head,
            reversible=self.params.reversible
        ).cuda()

        if self.params.use_horovod:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        self.optimizer = self.get_optimizer()

    def train(self, dataset):
        if self.params.use_horovod:
            sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
            data_loader = DataLoader(dataset,
                                     batch_size=self.params.batch_size,
                                     sampler=sampler,
                                     num_workers=self.params.num_workers)
        else:
            data_loader = DataLoader(dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers)

        step = 0
        exp_config = vars(self.params)
        exp_config['num_params'] = sum(p.numel() for p in self.model.parameters())
        if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
            run = wandb.init(
                project='storyteller',
                name=self.params.experiment_dpath.split('/')[-1],
                config=exp_config
            )

        for epoch in range(self.params.epochs):
            if self.params.use_horovod:
                sampler.set_epoch(epoch)
            logging.info(f'Starting epoch {epoch}')
            for i, data in enumerate(data_loader):
                images = data['image'].cuda()
                loss, recons = self.model(images, return_loss=True, return_recons=True, temp=temp)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                    logs = self.log_train(loss, recons, images, i, epoch)
                    wandb.log(logs)

                step += 1
            logging.info(f'Finished epoch {epoch}')

        if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
            wandb.save(f'{self.params.experiment_dpath}/*')

        wandb.finish()

    def get_optimizer(self):
        if self.params.use_horovod:
            optimizer = Adam(self.model.parameters(), lr=hvd.size()*self.params.learning_rate)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=self.model.named_parameters(),
                                                 compression=hvd.Compression.fp16,
                                                 op=hvd.Average,
                                                 gradient_predivide_factor=1.0)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.params.learning_rate)

        return optimizer

    def log_train(self, loss, recons, train_images, step, epoch):
        logs = {}
        if step % self.params.log_tier2_interval == 0:
            with torch.no_grad():
                codes = self.model.get_codebook_indices(train_images[:self.params.num_images_save])
                hard_recons = self.model.decode(codes)
            images, recons = train_images[:self.params.num_images_save], recons[:self.params.num_images_save]
            images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
            images, recons, hard_recons = map(
                lambda t: make_grid(t.float(), nrow=int(math.sqrt(self.params.num_images_save)), normalize=True, range=(-1, 1)),
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