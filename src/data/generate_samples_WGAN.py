from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
from tqdm import tqdm

import models.dcgan as dcgan
import models.mlp as mlp


def generate_samples(args):

    set_seed = 42
    random.seed(set_seed)
    torch.manual_seed(set_seed)
    torch.cuda.manual_seed(set_seed)
    torch.cuda.manual_seed_all(set_seed)

    #with open(args.params_path, 'r') as gencfg:
    #    generator_config = json.loads(gencfg.read())
    
    generator_config = vars(args)
    imageSize = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    noBN = generator_config["noBN"]
    ngpu = generator_config["ngpu"]
    mlp_G = generator_config["mlp_G"]
    n_extra_layers = generator_config["n_extra_layers"]


    if noBN:
        netG = dcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif mlp_G:
        netG = mlp.MLP_G(imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)


    # initialize noise
    fixed_noise = torch.FloatTensor(args.num_samples, nz, 1, 1).normal_(0, 1)

    if args.cuda:
        netG.load_state_dict(torch.load(args.weights_path)['netG_state_dict'])
        netG.cuda()
        fixed_noise = fixed_noise.cuda()
    else:
        netG.load_state_dict(torch.load(args.weights_path, map_location=torch.device('cpu'))['netG_state_dict'])


    fake = netG(fixed_noise)
    fake.data = fake.data.mul(0.5).add(0.5)

    folder_name = args.weights_path.parent.name
    save_dir = str(args.output_dir / f"wgan_{folder_name}")
    print(f"Saving on {save_dir}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    for i in tqdm(range(args.num_samples)):
        vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(save_dir, "generated_%02d.png"%i))



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument('-c', '--params_path', required=True, type=str, help='path to generator config .json file')
    #parser.add_argument('-w', '--weights_path', required=True, type=str, help='path to generator weights .pth file')
    #parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
    parser.add_argument('-n', '--num_samples', type=int, help="number of images to generate", default=1)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    args = parser.parse_args()

    generate_samples(args)
