# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license, 
# visit https://opensource.org/licenses/MIT.

"""Train a SaGAN"""

import argparse
import datetime
import itertools
import json
import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
from torchsummary import summary
from tensorboardX import SummaryWriter

from data import CelebA
from sagan import Generator, Discriminator


# Default CelebA 40 attributes
celeba_attrs = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heaay_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
    'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

def loop(iterator):
    while True:
        for item in iterator:
            yield item

def trainable(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)

def init_weights(m):
    if type(m) is nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        m.bias.data.fill_(0.0)
    elif type(m) is nn.Conv2d:
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        m.bias.data.fill_(0.0)
    elif type(m) is nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        m.bias.data.fill_(0.0)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/celeba')
    parser.add_argument('--attr-path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--target-attr', type=str, choices=celeba_attrs, required=True)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--l1', type=float, default=20)
    parser.add_argument('--l2', type=float, default=100)
    parser.add_argument('--lgp', type=float, default=10)
    parser.add_argument('--d-iters', type=int, default=3)
    parser.add_argument('--total-kimg', type=int, default=1000)
    parser.add_argument('--tick-kimg', type=float, default=5.0)
    parser.add_argument('--sample-ticks', type=int, default=1)
    parser.add_argument('--save-ticks', type=int, default=10)
    parser.add_argument('--num-samples', type=int, default=64)
    parser.add_argument('--experiment-name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%dM%H:%M.%f"))
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    # Arguments
    args = parse()
    print(args)
    
    # Device
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    
    # Paths
    checkpoint_path = join('results', args.experiment_name, 'checkpoint')
    sample_path = join('results', args.experiment_name, 'sample')
    summary_path = join('results', args.experiment_name, 'summary')
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(summary_path, exist_ok=True)
    with open(join('results', args.experiment_name, 'setting.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    writer = SummaryWriter(summary_path)
    
    # Data
    selected_attrs = [args.target_attr]
    train_dset = CelebA(args.data_path, args.attr_path, args.image_size, 'train', selected_attrs)
    train_data = data.DataLoader(train_dset, args.batch_size, shuffle=True, drop_last=True)
    train_data = loop(train_data)
    test_dset = CelebA(args.data_path, args.attr_path, args.image_size, 'test', selected_attrs)
    test_data = data.DataLoader(test_dset, args.num_samples)
    for fixed_reals, fixed_labels in test_data:
        # Get the first batch of images from the testing set
        fixed_reals, fixed_labels = fixed_reals.to(device), fixed_labels.type_as(fixed_reals).to(device)
        fixed_target_labels = 1 - fixed_labels
        break
    del test_dset
    del test_data
    vutils.save_image(fixed_reals, join(sample_path, '{:07d}_real.jpg'.format(0)), nrow=8, padding=0, normalize=True, value_range=(-1., 1.))
    
    # Models
    G = Generator()
    G.apply(init_weights)
    G.to(device)
    
    D = Discriminator()
    D.apply(init_weights)
    D.to(device)
    
    # Optimizers
    G_opt = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    D_opt = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    cross_entropy = torch.nn.BCELoss()
    l1_norm = torch.nn.L1Loss()
    
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    while cur_nimg < args.total_kimg * 1000:
        G.train()
        D.train()
        for _ in range(args.d_iters):
            # Train D
            trainable(G, False)
            trainable(D, True)
            
            reals, labels = next(train_data)
            reals, labels = reals.to(device), labels.type_as(reals).to(device)
            target_labels = 1 - labels
            
            fakes, _ = G(reals, target_labels)
            fakes = fakes.detach()
            d_real, dc_real = D(reals)
            d_fake, dc_fake = D(fakes)
            
            df_loss = d_fake.mean() - d_real.mean()
            dc_loss = cross_entropy(dc_real, labels)
            alpha = torch.rand(args.batch_size, 1, 1, 1).to(device)
            mix_in = (1-alpha) * reals + alpha * fakes
            mix_in.requires_grad = True
            mix_out, _ = D(mix_in)
            grad = torch.autograd.grad(
                outputs=mix_out, inputs=mix_in, 
                grad_outputs=torch.ones_like(mix_out), 
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            df_gp = ((norm - 1.0) ** 2).mean()
            
            d_loss = df_loss + dc_loss + args.lgp * df_gp
            D_opt.zero_grad()
            d_loss.backward()
            D_opt.step()
            
            cur_nimg += args.batch_size
        
        # Train G
        trainable(G, True)
        trainable(D, False)
        
        reals, labels = next(train_data)
        reals, labels = reals.to(device), labels.type_as(reals).to(device)
        target_labels = 1 - labels
        
        fakes, _ = G(reals, target_labels)
        fakes_crec, _ = G(fakes, labels)
        fakes_srec, _ = G(reals, labels)
        d_fake, dc_fake = D(fakes)
        
        gf_loss = -d_fake.mean()
        gc_loss = cross_entropy(dc_fake, target_labels)
        gr_loss = args.l1 * l1_norm(fakes_crec, reals) + args.l2 * l1_norm(fakes_srec, reals)
        
        g_loss = gf_loss + gc_loss + gr_loss
        G_opt.zero_grad()
        g_loss.backward()
        G_opt.step()
        
        done = (cur_nimg >= args.total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + args.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_start_nimg = cur_nimg
            
            # Training log
            print('kimg {:.1f} | d_loss {:.6f} g_loss {:.6f} | df {:.6f} dc {:.6f} gp {:.6f} gf {:.6f} gc {:.6f} gr {:.6f}'.format(cur_nimg / 1000, d_loss.item(), g_loss.item(), df_loss.item(), dc_loss.item(), df_gp.item(), gf_loss.item(), gc_loss.item(), gr_loss.item()))
            add_scalar_dict(writer, {
                'kimg': cur_nimg / 1000
            }, cur_nimg, 'Progress')
            add_scalar_dict(writer, {
                'd_loss': d_loss.item(),
                'df_loss': df_loss.item(),
                'dc_loss': dc_loss.item(),
                'df_gp': df_gp.item()
            }, cur_nimg, 'D')
            add_scalar_dict(writer, {
                'g_loss': g_loss.item(),
                'gf_loss': gf_loss.item(),
                'gc_loss': gc_loss.item(),
                'gr_loss': gr_loss.item()
            }, cur_nimg, 'G')
            
            # Training samples
            if cur_tick % args.sample_ticks == 0 or done:
                G.eval()
                with torch.no_grad():
                    samples, masks = G(fixed_reals, fixed_target_labels)
                    vutils.save_image(samples, join(sample_path, '{:07d}_fake.jpg'.format(cur_nimg)), nrow=8, padding=0, normalize=True, value_range=(-1., 1.))
                    vutils.save_image(masks.repeat(1, 3, 1, 1), join(sample_path, '{:07d}_mask.jpg'.format(cur_nimg)), nrow=8, padding=0)
            
            # Model checkpoints
            if cur_tick % args.save_ticks == 0 or done:
                torch.save(G.state_dict(), join(checkpoint_path, '{:07}.G.pth'.format(cur_nimg)))
                torch.save(D.state_dict(), join(checkpoint_path, '{:07}.D.pth'.format(cur_nimg)))
                torch.save(G_opt.state_dict(), join(checkpoint_path, '{:07}.G_opt.pth'.format(cur_nimg)))
                torch.save(D_opt.state_dict(), join(checkpoint_path, '{:07}.D_opt.pth'.format(cur_nimg)))
