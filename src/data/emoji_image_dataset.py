import argparse
import os
from argparse import Namespace

import pandas as pd
import torch
from PIL import Image
from dalle_pytorch.tokenizer import tokenizer
from torch.utils.data import Dataset
from torchvision import transforms as T


class EmojiImage(Dataset):
    def __init__(self, *args):
        if len(args) == 2:
            generic_args, other_args = args
            self.parser = argparse.ArgumentParser()
            if generic_args.which == 'train':
                self.parser.add_argument('--dpath', type=str, required=True)
                self.parser.add_argument('--image_size', type=int, required=True)
                self.parser.add_argument('--text_seq_len', type=int, required=True)
            self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
            self.params.vocab_size = tokenizer.vocab_size
            self._initialize()
        elif isinstance(args[0], Namespace):
            self.params = args[0]
            self.params.vocab_size = tokenizer.vocab_size
            self._initialize()

    def _initialize(self):
        self.dataset = pd.read_json(os.path.join(self.params.dpath, 'processed.json'),
                                    orient='records', lines=True).reset_index()
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

    def tokenize(self, text):
        return tokenizer.tokenize(text,
                                  self.params.text_seq_len)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        caption_txt = self.dataset.iloc[idx, 1]
        img_fpath = os.path.join(self.params.dpath, self.dataset.iloc[idx, 2])

        with open(img_fpath, 'rb') as fpath:
            img = Image.open(fpath)
            img = self.transform(img)
            caption = self.tokenize(caption_txt).squeeze(0)

        return {
            'image': img,
            'caption': caption,
            'mask': caption != 0,
            'text': caption_txt
        }