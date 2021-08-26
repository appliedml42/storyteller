import argparse
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ConceptualCaptions(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, generic_args, other_args):
        self.parser = argparse.ArgumentParser()
        if generic_args.which == 'train':
            self.parser.add_argument('--dpath', type=str, required=True)
            self.parser.add_argument('--image_size', type=int, required=True)
            self.parser.add_argument('--caption_mapping_fpath', type=str, required=True)
        self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
        # with open(self.params.caption_mapping_fpath) as reader:
        #   self.mapping = {fpath:caption for fpath, caption in json.load(reader)}
        self.fpaths = [os.path.join(self.params.dpath, x) for x in os.listdir(self.params.dpath)]
        self.image_size = self.params.image_size
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fpath = self.fpaths[idx]
        # caption = self.mapping[self.fpaths[idx]]
        with open(fpath, 'rb') as fpath:
            img = Image.open(fpath)
            img = self.transform(img)
        return {
            'image': img,
            'caption': 'tbd'
        }
