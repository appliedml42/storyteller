import argparse
from argparse import Namespace
import logging
import os

import pandas as pd
import requests
import torch
from PIL import Image
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
            self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
            self._initialize()
        elif isinstance(args[0], Namespace):
            self.params = args[0]
            self._initialize()

    def _initialize(self):
        self.dataset = pd.read_csv(os.path.join(self.params.fpath, 'Train_GCC-training.tsv'),
                                   sep='\t',
                                   names=['caption', 'url'],
                                   header=None)
        self.image_size = self.params.image_size
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
        try:
            with requests.get(url, timeout=2, stream=True) as response:
                if response.status_code == 200:
                    return Image.open(bytearray(response.content))
                else:
                    return None
        except Exception as e:
            logging.error(e)
            return None

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        caption = self.dataset.iloc[:, idx[0]]
        img_url = self.dataset.iloc[:, idx[1]]
        img = ConceptualCaptionsWeb._download_image(img_url)

        # If retrieval of image from original index failed replace with random sample.
        while img is None:
            img = ConceptualCaptionsWeb._download_image(img_url)
            rand_sample = self.dataset.sample()
            caption = rand_sample['caption']
            img_url = rand_sample['url']

        img = self.transform(img)
        return {
            'image': img,
            'caption': caption
        }


ds = ConceptualCaptionsWeb