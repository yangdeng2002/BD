import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from advertorch.attacks import LinfPGDAttack,MomentumIterativeAttack,LinfBasicIterativeAttack
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
from model.resnet import *
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.485,0.456,0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229,0.224,0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument('--step_size', default=2, type=float, help='perturb step size')
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")


opt = parser.parse_args()


def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


def main():
    
    '''create model here'''
    bdres=resnet50(pretrained=True, pool_only=False, replace_stride_with_dilation=[False,2,2])
    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),bdres.eval().cuda()
            #pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda()
                               )
    
    
    transforms = T.Compose([T.Resize(224), T.ToTensor()])
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    
    epsilon = opt.max_epsilon / 255.0
    if opt.step_size < 0:
        step_size = epsilon / opt.num_iter_set
    else:
        step_size = opt.step_size / 255.0
    print('max epsilon = {}'.format(epsilon))
    print('iterative step_size = {}'.format(step_size))
    print('using momentum iteartive attack with momentum = {}'.format(opt.momentum))
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()

        
        adversary = MomentumIterativeAttack(predict=model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                            eps=epsilon, nb_iter=opt.num_iter_set, eps_iter=step_size,
                                            decay_factor=opt.momentum,
                                            clip_min=0.0, clip_max=1.0, targeted=False)
        
        adv_img = adversary.perturb(images, gt)

        adv_img_np = np.transpose(adv_img.cpu().numpy(), (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()