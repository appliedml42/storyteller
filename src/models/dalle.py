import argparse
import json
import logging
import time
from abc import ABC
from argparse import Namespace

import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dalle_pytorch import DALLE as DALLE_MODEL
from dalle_pytorch import VQGanVAE
from einops import repeat
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import horovod.torch as hvd
from common.utils import get_model_class, get_dataset_class, generate_images


class DALLE(ABC):
    def __init__(self, generic_args, other_args):

        self.parser = argparse.ArgumentParser()
        if generic_args.which == 'train':
            self.parser.add_argument('--vae_config_fpath', type=str, required=False, default=None)
            self.parser.add_argument('--vae_weights_fpath', type=str, required=False, default=None)
            self.parser.add_argument('--vae_type', type=str, required=True, choices=['vqgan', 'vqvae'])
            self.parser.add_argument('--batch_size', type=int, required=True)
            self.parser.add_argument('--learning_rate', type=float, required=True)
            self.parser.add_argument('--dim', type=int, required=True)
            self.parser.add_argument('--depth', type=int, required=True)
            self.parser.add_argument('--heads', type=int, required=True)
            self.parser.add_argument('--dim_head', type=int, required=True)
            self.parser.add_argument('--reversible', default=False, action='store_true')
            self.parser.add_argument('--clip_grad_norm', type=float, required=False, default=None)
            self.parser.add_argument('--num_workers', type=int, required=True)
            self.parser.add_argument('--prefetch_factor', type=int, required=False, default=2)
            self.parser.add_argument('--cache_duration', type=int, required=False)
            self.parser.add_argument('--attn_types', type=str, required=True)
            self.parser.add_argument('--loss_img_weight', type=int, required=True)
            self.parser.add_argument('--log_tier1_interval', type=int, required=True)
            self.parser.add_argument('--log_tier2_interval', type=int, required=True)
            self.parser.add_argument('--save_interval', type=int, required=True)
            self.parser.add_argument('--weights_fpath', type=str, required=False, default=None)
            self.parser.add_argument('--lr_schedule', choices=['rlop', 'nos'], required=True, type=str)
            self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
        elif generic_args.which == 'use' and generic_args.method == 'generate_images':
            self.parser.add_argument('--prompt', type=str, required=True)
            self.parser.add_argument('--weights_fpath', type=str, required=True)
            self.parser.add_argument('--num_images', type=int, required=True)
            self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
            self.params.use_horovod = False

        if self.params.vae_type == 'vqgan':
            self.vae = VQGanVAE(vqgan_model_path=self.params.vae_weights_fpath,
                                vqgan_config_path=self.params.vae_config_fpath)
        elif self.params.vae_type == 'vqvae':
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
            reversible=self.params.reversible,
            loss_img_weight=self.params.loss_img_weight,
            attn_types=tuple(self.params.attn_types.split(',')),
            rotary_emb=False,
            shift_tokens=False
        ).cuda()

        if self.params.use_horovod:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        self.optimizer, self.scheduler = self.get_optimizer()

        if 'weights_fpath' in vars(self.params) and self.params.weights_fpath is not None:
            weights = torch.load(self.params.weights_fpath)['weights']
            self.model.load_state_dict(weights)

    def train(self, dataset):
        sampler = None
        shuffle = True
        if self.params.use_horovod:
            sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
            shuffle = False

        data_loader = DataLoader(dataset,
                                 batch_size=self.params.batch_size,
                                 sampler=sampler,
                                 num_workers=self.params.num_workers,
                                 prefetch_factor=self.params.prefetch_factor,
                                 shuffle=shuffle)

        step = 0
        exp_config = vars(self.params)
        exp_config['num_params'] = sum(p.numel() for p in self.model.parameters())
        exp_config['num_trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
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
                self.optimizer.zero_grad()
                if self.params.cache_duration is not None and not cached:
                    logging.info('Caching data...')
                    time.sleep(self.params.cache_duration)
                    cached = True
                    logging.info('Caching data done')
                images = data['image'].cuda()
                captions = data['caption'].cuda()

                loss = self.model(captions,
                                  images,
                                  return_loss=True)
                loss.backward()
                self.optimizer.step()

                if self.params.clip_grad_norm is not None:
                    clip_grad_norm(self.model.parameters(), self.params.clip_grad_norm)

                if self.params.use_horovod:
                    loss = hvd.allreduce(loss).item()

                self.log_train(loss, images, captions, dataset.tokenizer, i, step, epoch)
                step += 1
            logging.info(f'Finished epoch {epoch}')
            if self.scheduler is not None:
                self.scheduler.step(loss)
            if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                save_obj = {
                    'weights': self.model.state_dict()
                }
                torch.save(save_obj, f'{self.params.experiment_dpath}/dalle.pt')
                if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                    wandb.save(f'{self.params.experiment_dpath}/*')
            cached = False
        wandb.finish()

    def get_optimizer(self):
        if self.params.use_horovod:
            optimizer = Adam(self.model.parameters(), lr=self.params.learning_rate * hvd.size())
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=self.model.named_parameters(),
                                                 compression=hvd.Compression.fp16,
                                                 backward_passes_per_step=5,
                                                 op=hvd.Average)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.params.learning_rate)

        scheduler = None
        if self.params.lr_schedule == 'rlop':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                cooldown=10,
                min_lr=1e-6,
                verbose=True,
            )

        return optimizer, scheduler

    def log_train(self, loss, train_images, train_texts, tokenizer, local_step, global_step, epoch):
        logs = {}
        if global_step != 0 and global_step % self.params.log_tier2_interval == 0:
            if self.params.cache_duration is not None:
                logging.info(f'Sleeping for {self.params.cache_duration} secs to cache data.')
                time.sleep(self.params.cache_duration)
                logging.info('Caching data done')

            if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                sample_text = train_texts[:1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list, pad_tokens=set())
                image = train_images[:1]

                with torch.no_grad():
                    vae_codes = self.model.vae.get_codebook_indices(train_images[:1])
                    vae_reconstruction = self.model.vae.decode(vae_codes)
                    dalle_reconstruction, dalle_codes = generate_images(sample_text, self.model, filter_thres=0.9)
                image, vae_codes, vae_reconstruction, dalle_reconstruction, dalle_codes = map(lambda t: t.detach().cpu(),
                                                                             (image,
                                                                              vae_codes,
                                                                              vae_reconstruction,
                                                                              dalle_reconstruction,
                                                                              dalle_codes))

                logs = {
                    **logs,
                    'vae_codebook_indices': wandb.Histogram(vae_codes),
                    'dalle_codebook_indices': wandb.Histogram(dalle_codes),
                    'orig_image': wandb.Image(image, caption=decoded_text),
                    'vae_recon': wandb.Image(vae_reconstruction, caption='VAE reconstruction'),
                    'dalle_recon': wandb.Image(dalle_reconstruction, caption='DALLE reconstruction')
                }

        if global_step != 0 and global_step % self.params.log_tier1_interval == 0:
            if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
                lr = self.optimizer.param_groups[0]['lr']
                logging.info(f'Epoch:{epoch} Step:{global_step} loss:{loss} lr:{lr}')
                logs = {
                    **logs,
                    'epoch': epoch,
                    'step': local_step,
                    'loss': loss,
                    'lr': lr
                }

        if (self.params.use_horovod and hvd.rank() == 0) or not self.params.use_horovod:
            wandb.log(logs)

    def generate_images(self):
        dataset_class = get_dataset_class(self.params)
        dataset = dataset_class(self.params)
        text = dataset.tokenize(self.params.prompt).cuda()
        text = repeat(text, '() n -> b n', b=self.params.num_images)
        images = self.model.generate_images(
            text,
            filter_thres=0.9
        )
        images = make_grid(images, nrow=int(self.params.num_images / 5))
        save_image(images, f'{"_".join(self.params.prompt.lower().split())}.jpeg')
