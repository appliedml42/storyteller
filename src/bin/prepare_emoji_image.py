import argparse
import logging
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import requests

logging.getLogger().addHandler(logging.StreamHandler())


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath', required=True)
    return parser


def get_image(args):
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

    dataset = pd.read_csv(os.path.join(cmd.dpath, 'full_emoji.csv'))

    emoji_group = ['Apple',
                   'Google',
                   'Facebook',
                   'Windows',
                   'Twitter',
                   'JoyPixels',
                   'Samsung',
                   'Gmail',
                   'SoftBank',
                   'DoCoMo',
                   'KDDI']

    output_rows = []
    for index, row in dataset.iterrows():
        for eg in emoji_group:
            if not pd.isnull(row[eg]):
                image_fpath = os.path.join(cmd.dpath,'image', eg, f'{index + 1}.png')
                if not os.path.isfile(image_fpath):
                    raise ValueError(f'Image not found for {image_fpath}')
                output_rows.append({
                    'caption':row['name'],
                    'image': os.path.join('image', eg, f'{index + 1}.png')
                })

    pd.DataFrame(output_rows).to_json(os.path.join(cmd.dpath, 'processed.json'),
                                      lines=True,
                                      orient='records')

