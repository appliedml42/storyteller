import argparse
import logging
import os
from argparse import Namespace

import pandas as pd
import requests
import torch
from PIL import Image
from dalle_pytorch.tokenizer import tokenizer
from torch.utils.data import Dataset
from torchvision import transforms as T


class ConceptualCaptionsWeb(Dataset):
    def __init__(self, *args):
        if len(args) == 2:
            generic_args, other_args = args
            self.parser = argparse.ArgumentParser()
            if generic_args.which == 'train':
                self.parser.add_argument('--dpath', type=str, required=True)
                self.parser.add_argument('--image_size', type=int, required=True)
                self.parser.add_argument('--text_seq_len', type=int, required=True)
                self.parser.add_argument('--gen_logs', action='store_true', default=False)
            self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
            self.params.vocab_size = tokenizer.vocab_size
            self._initialize()
        elif isinstance(args[0], Namespace):
            self.params = args[0]
            self._initialize()

    def _initialize(self):
        self.dataset = pd.read_csv(os.path.join(self.params.dpath, 'Train_GCC-training.tsv'),
                                   sep='\t',
                                   names=['caption', 'url'],
                                   header=None).reset_index()
        self.image_size = self.params.image_size
        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _download_image(url):
        with requests.get(url, timeout=2, stream=True) as response:
            if response.status_code == 200:
                return Image.open(response.raw)
            else:
                return None

    def tokenize(self, text):
        return tokenizer.tokenize(text,
                                  self.params.text_seq_len)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = None
        caption = self.dataset.iloc[idx, 1]
        img_url = self.dataset.iloc[idx, 2]
        while img is None:
            try:
                img = ConceptualCaptionsWeb._download_image(img_url)
                img = self.transform(img)
                caption = self.tokenize(caption).squeeze(0)
            except Exception as e:
                img = None
                rand_sample = self.dataset.sample()
                caption = rand_sample['caption'].values[0]
                img_url = rand_sample['url'].values[0]
                if self.params.gen_logs:
                    logging.error(e)

        if self.params.gen_logs:
            logging.info(f'idx:{idx} url:{img_url}')

        return {
            'image': img,
            'caption': caption,
            'mask': caption != 0
        }