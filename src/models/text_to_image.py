import argparse
import json
from argparse import Namespace

import einops
import pytorch_lightning as plm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import wandb
from common.utils import get_model_class
from models.optimizer_modules import CosineWarmupScheduler
from transformer_modules import Embedding, Encoder, Decoder
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

class TextToImage(plm.LightningModule):

    def __init__(self,
                 experiment_dpath,
                 num_epochs,
                 d_model,
                 vocab_size,
                 num_layers,
                 num_heads,
                 dropout,
                 vae_config_fpath,
                 text_seq_len,
                 vae_weights_fpath, **kwargs):

        super(TextToImage, self).__init__()
        self.save_hyperparameters()

        with open(self.hparams.vae_config_fpath) as reader:
            experiment_configuration = Namespace(**json.load(reader))
            model_class = get_model_class(experiment_configuration, False)
            experiment_configuration.use_horovod = False
            model = model_class(experiment_configuration)
            self.vae = model.model
            weights = torch.load(self.hparams.vae_weights_fpath)['weights']
            self.vae.load_state_dict(weights)
            self.hparams.num_codebook_tokens = self.vae.num_tokens
            image_fmap_size = (self.vae.image_size // (2 ** self.vae.num_layers))
            self.hparams.image_seq_length = image_fmap_size ** 2
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.cuda()

        self.text_embeddings = Embedding(self.hparams.d_model,
                                         self.hparams.vocab_size,
                                         self.hparams.text_seq_len,
                                         enable_padding=True)

        self.image_embeddings = Embedding(self.hparams.d_model,
                                          self.hparams.num_codebook_tokens + 1,
                                          self.hparams.image_seq_length)

        self.encoder = Encoder(self.hparams.num_layers,
                               self.hparams.num_heads,
                               self.hparams.d_model,
                               self.hparams.dropout)

        self.decoder = Decoder(self.hparams.num_layers,
                               self.hparams.num_heads,
                               self.hparams.d_model,
                               self.hparams.image_seq_length,
                               self.hparams.dropout)

        self.codebook_layer = nn.Linear(self.hparams.d_model, self.hparams.num_codebook_tokens, bias=False)
        self.accuracy = Accuracy()
        self.loss_module = nn.NLLLoss()

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group('TextToImage')
        if parent_parser.which == 'train':
            parser.add_argument('--d_model', type=int, required=True)
            parser.add_argument('--vocab_size', type=int, required=True)
            parser.add_argument('--num_layers', type=int, required=True)
            parser.add_argument('--num_heads', type=int, required=True)
            parser.add_argument('--dropout', type=int, required=True)
            parser.add_argument('--vae_config_fpath', type=str, required=True)
            parser.add_argument('--vae_weights_fpath', type=str, required=True)
        return parent_parser

    def forward(self, image_input, caption_input, caption_mask):
        caption_output = self.text_embeddings(caption_input)
        caption_output = self.encoder(caption_output, caption_mask)

        image_output = self.image_embeddings(image_input)
        image_output = self.decoder(caption_output,
                                    image_output,
                                    encoder_padding_mask=caption_mask,
                                    decoder_padding_mask=None)
        image_output = self.codebook_layer(image_output)
        image_output = F.log_softmax(image_output, dim=-1)

        return image_output

    def training_step(self, batch, batch_idx):
        image_input, caption_input, caption_mask = batch['image'], batch['caption'].int(), batch['mask']
        image_input = self.vae.get_codebook_indices(image_input)
        shifted_image_input = torch.cat([torch.zeros(image_input.size(0), 1).to(self.device), image_input[:, :-1] + 1],
                                        dim=1).int()
        image_output = self.forward(shifted_image_input, caption_input, caption_mask)
        image_input = einops.rearrange(image_input,
                                       'batch_size seq_len -> (batch_size seq_len)')
        image_output = einops.rearrange(image_output,
                                        'batch_size seq_len class -> (batch_size seq_len) class')
        loss = F.nll_loss(image_output, image_input)

        self.log('train_loss', loss)
        self.log('train_accuracy', self.accuracy(image_output, image_input))

        return loss

    def validation_step(self, batch, batch_idx):
        image_true, caption_input, caption_mask, caption_text = batch['image'], \
                                                                batch['caption'].int(), \
                                                                batch['mask'], \
                                                                batch['text']
        pred_codes = torch.zeros(1, self.hparams.image_seq_length, device=self.device, dtype=torch.int)
        image_input = torch.zeros(1, self.hparams.image_seq_length, device=self.device, dtype=torch.int)
        true_codes = self.vae.get_codebook_indices(image_true)

        for i in range(self.hparams.image_seq_length):
            model_output = torch.exp(self.forward(image_input, caption_input, caption_mask))
            model_output = model_output[0, i, :]
            logits, indices = torch.topk(model_output, 1)
            pred_codes[0, i] = indices[0]
            if i < self.hparams.image_seq_length - 1:
                image_input[0, i + 1] = pred_codes[0, i] + 1

        model_image = self.vae.decode(pred_codes).detach()[0]
        vae_image = self.vae.decode(true_codes).detach()[0]

        self.wandb_logger.log_image(key='Images',
                                    images=[image_true, model_image, vae_image],
                                    caption=[f'True {caption_text[0]}',
                                             f'Model {caption_text[0]}',
                                             f'VAE {caption_text[0]}'
                                             ])

        wandb.log({'model codes': wandb.Histogram(pred_codes.detach().cpu()),
                   'vae codes': wandb.Histogram(true_codes.detach().cpu())})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=self.hparams.num_epochs * 224)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_model(self, train_dataloader, val_dataloader, test_dataloader):
        logger = WandbLogger(project='storyteller_v2',
                             name=self.hparams.experiment_dpath.split('/')[-1],
                             save_dir=self.hparams.experiment_dpath)

        self.wandb_logger = logger
        self.wandb_logger.watch(self, log_graph=True)

        lr_monitor = plm.callbacks.LearningRateMonitor(logging_interval='step')
        model_checkpoint = plm.callbacks.ModelCheckpoint(dirpath=self.hparams.experiment_dpath)

        trainer = plm.Trainer(
            max_epochs=self.hparams.num_epochs,
            enable_progress_bar=False,
            gpus=1,
            callbacks=[lr_monitor, model_checkpoint],
            val_check_interval=50,
            log_every_n_steps=1,
            logger=logger,
            gradient_clip_val=0.5
        )

        trainer.fit(self,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)


from data.emoji_image_dataset import EmojiImage
import names
import os

run_name = '_'.join(names.get_full_name().lower().split())
experiment_dpath = os.path.join('/workspace/experiment/storyteller_v2', run_name)
os.makedirs(experiment_dpath)

dataset_params = Namespace()
dataset_params.dpath = '/workspace/data/storyteller/emoji_images'
dataset_params.image_size = 32
dataset_params.text_seq_len = 100

dataset = EmojiImage(dataset_params)

model = TextToImage(
    experiment_dpath=experiment_dpath,
    num_epochs=180,
    d_model=1024,
    vocab_size=dataset.params.vocab_size,
    num_layers=2,
    num_heads=8,
    dropout=0.1,
    text_seq_len=dataset.params.text_seq_len,
    vae_config_fpath='/workspace/experiment/storyteller/vqvae/emoji/Judy_Parrino/config.json',
    vae_weights_fpath='/workspace/experiment/storyteller/vqvae/emoji/Judy_Parrino/model_500.pt'
)

train_dataloader = DataLoader(dataset,
                              batch_size=64,
                              num_workers=12)

val_dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=1,
                            sampler=RandomSampler(dataset, num_samples=1, replacement=True))

model.train_model(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  test_dataloader=None)
