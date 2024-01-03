# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license, 
# visit https://opensource.org/licenses/MIT.

"""Generate images from trained models"""

import argparse
import json
import os
from os import listdir
from os.path import join
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from data import CelebA
from sagan import Generator


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--attr-path', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--batch-size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--test-nimg', type=int, default=None)
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    # Arguments
    args = parse()
    print(args)
    
    # Load training setting
    with open(join('results', args.experiment_name, 'setting.json'), 'r', encoding='utf-8') as f:
        setting = json.load(f)
    for key, value in vars(args).items():
        setting[key] = value
    args = argparse.Namespace(**setting)
    print(args)
    
    # Device
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    
    # Paths
    checkpoint_path = join('results', args.experiment_name, 'checkpoint')
    test_path = join('results', args.experiment_name, 'test')
    os.makedirs(test_path, exist_ok=True)
    
    # Data
    selected_attrs = [args.target_attr]
    test_dset = CelebA(args.data_path, args.attr_path, args.image_size, 'test', selected_attrs)
    test_data = data.DataLoader(test_dset, args.batch_size)
    
    # Model
    G = Generator()
    G.to(device)
    
    # Load from checkpoints
    load_nimg = args.test_nimg
    if load_nimg is None:  # Use the lastest model
        load_nimg = max(int(path.split('.')[0]) for path in listdir(join(checkpoint_path)) if path.split('.')[0].isdigit())
    print('Loading generator from nimg {:07d}'.format(load_nimg))
    G.load_state_dict(torch.load(
        join(checkpoint_path, '{:07d}.G.pth'.format(load_nimg)),
        map_location=lambda storage, loc: storage
    ))
    
    G.eval()
    with torch.no_grad():
        for batch_idx, (reals, labels) in enumerate(tqdm(test_data)):
            reals, labels = reals.to(device), labels.type_as(reals).to(device)
            target_labels = 1 - labels
            
            # Modify images
            samples, masks = G(reals, target_labels)
            
            # Put images together
            masks = masks.repeat(1, 3, 1, 1) * 2 - 1
            images_out = torch.stack((reals, samples, masks))  # 3, N, 3, S, S
            images_out = images_out.transpose(0, 1)  # N, 3, 3, S, S
            
            # Save images separately
            for idx, image_out in enumerate(images_out):
                vutils.save_image(
                    image_out,
                    join(test_path, '{:06d}.jpg'.format(batch_idx*args.batch_size+idx+200000)),
                    nrow=3,
                    padding=0,
                    normalize=True,
                    range=(-1.,1.)
                )
