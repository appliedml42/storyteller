import argparse
import json
import logging
import os
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import requests

logging.getLogger().addHandler(logging.StreamHandler())


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath', required=True)
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    cmd, other_args = parser.parse_known_args()

    bird_images = []
    rows = []
    for dpath in os.listdir(os.path.join(cmd.dpath, 'images')):
        image_dpath = os.path.join(cmd.dpath, 'images', dpath)
        caption_dpath = os.path.join(cmd.dpath, 'text', dpath)
        image_fpaths = [os.path.join(image_dpath, x) for x in os.listdir(image_dpath)]
        caption_fpaths = [os.path.join(caption_dpath, x) for x in os.listdir(caption_dpath)]
        for image_fpath, caption_fpath in zip(image_fpaths, caption_fpaths):
            with open(caption_fpath) as reader:
                captions = [x.lower().strip() for x in reader.readlines()]
            for caption in captions:
                rows.append({
                    'image_fpath': image_fpath,
                    'caption': caption,
                    'bird': dpath
                })

    df = pd.DataFrame(rows)
    df.to_json(os.path.join(cmd.dpath, 'df.json'), lines=True, orient='records')
    with open(os.path.join(cmd.dpath, 'raw.txt'), 'w') as writer:
        for caption in df.caption.tolist():
            writer.write(f'{caption}\n')