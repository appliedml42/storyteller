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


def download_image(args):
    try:
        caption, url, processed_image_dpath = args
        fpath = tempfile.mktemp(dir=processed_image_dpath)
        with requests.get(url, timeout=2, stream=True) as response:
            if response.status_code == 200:
                arr = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(arr, -1)
                cv2.imwrite(f'{fpath}.jpeg', img)
    except Exception as e:
        logging.exception(e)
        return None

    return caption, fpath


if __name__ == '__main__':
    parser = get_args_parser()
    cmd, other_args = parser.parse_known_args()

    dataset = pd.read_csv(os.path.join(cmd.dpath, 'Train_GCC-training.tsv'),
                          sep='\t',
                          names=['caption', 'image_url'],
                          header=None)

    processed_image_dpath = tempfile.mkdtemp(prefix='Train_GCC_training_images',
                                             dir=cmd.dpath)

    orig_dataset = [(caption, image_url, processed_image_dpath) for caption, image_url in
                    zip(dataset.caption.tolist(), dataset.image_url.tolist())]

    pool = Pool(24)
    result = pool.map(download_image, orig_dataset)

    with open(os.path.join(cmd.dpath, 'image_map.json'), 'w') as writer:
        json.dump(result, writer)