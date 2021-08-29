import argparse
import json
import logging
import time
from abc import ABC
from argparse import Namespace

import horovod.torch as hvd
import torch
import wandb
from dalle_pytorch import DALLE as DALLE_MODEL
from dalle_pytorch import VQGanVAE
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam
from torch.utils.data import DataLoader

from common.utils import get_model_class


class DALLE(ABC):
    def __init__(self, generic_args, other_args):
        self.parser = argparse.ArgumentParser()
        if generic_args.which == 'train':
            self.parser.add_argument('--vae_config_fpath', type=str, required=False, default=None)
            self.parser.add_argument('--vae_weights_fpath', type=str, required=False, default=None)
            self.parser.add_argument('--batch_size', type=int, required=True)
            self.parser.add_argument('--learning_rate', type=float, required=True)
            self.parser.add_argument('--dim', type=int, required=True)
            self.parser.add_argument('--depth', type=int, required=True)
            self.parser.add_argument('--heads', type=int, required=True)
            self.parser.add_argument('--dim_head', type=int, required=True)
            self.parser.add_argument('--reversible', default=False, action='store_true')
            self.parser.add_argument('--clip_grad_norm', type=float, required=True)
            self.parser.add_argument('--num_workers', type=int, required=True)
            self.parser.add_argument('--prefetch_factor', type=int, required=True)
            self.parser.add_argument('--cache_duration', type=int, required=True)
            self.parser.add_argument('--log_tier1_interval', type=int, required=True)
            self.parser.add_argument('--log_tier2_interval', type=int, required=True)
            self.parser.add_argument('--save_interval', type=int, required=True)

        self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)

        if self.params.vae_config_fpath is None:
            self.vae = VQGanVAE()
        else:
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
            num_text_tokens=self.params.vocab_size,
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
                    logging.info('Caching data...')
                    time.sleep(self.params.cache_duration)
                    cached = True
                    logging.info('Caching data done')
                images = data['image'].cuda()
                captions = data['caption'].cuda()
                masks = data['mask'].cuda()
                loss = self.model(captions,
                                  images,
                                  mask=masks,
                                  return_loss=True)

                loss.backward()
                clip_grad_norm(self.model.parameters(), self.params.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                    logs = self.log_train(loss, captions, masks, dataset.tokenizer, i, epoch)
                    wandb.log(logs)

                step += 1
            logging.info(f'Finished epoch {epoch}')
            cached = False

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

    def log_train(self, loss, train_texts, train_masks, tokenizer, step, epoch):
        logs = {}
        if step != 0 and step % self.params.log_tier2_interval == 0:
            sample_text = train_texts[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            with torch.no_grad():
                image = self.model.generate_images(
                    sample_text,
                    mask=train_masks[:1],
                    filter_thres=0.9
                )

            logs = {
                **logs,
                'image': wandb.Image(image, caption=decoded_text)
            }

        if step != 0 and step % self.params.log_tier1_interval == 0:
            logging.info(f'Epoch:{epoch} Step:{step} loss:{loss.item()}')
            logs = {
                **logs,
                'epoch': epoch,
                'step': step,
                'loss': loss.item()
            }

        if step != 0 and step % self.params.save_interval == 0:
            if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                save_obj = {
                    'weights': self.model.state_dict()
                }
                torch.save(save_obj, f'{self.params.experiment_dpath}/vae_{epoch}_{step}.pt')

        return logs