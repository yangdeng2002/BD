"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import pretrainedmodels
import torchvision
from model.resnet import *
batch_size = 10

input_csv = './dataset/images.csv'
input_dir = './dataset/images'
adv_dir = './outputs'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def verify(model_name,model=None):
    if model_name!=None:
        net = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        height, width = net.input_size[1], net.input_size[2]
        model = nn.Sequential(Normalize(mean=net.mean, std=net.std), net.eval())
    else:
        model= nn.Sequential(Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), model.eval())
        height=224


    X = ImageNet(adv_dir, input_csv, T.Compose([T.Resize(height),T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    model=model.cuda()
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            pred=model(images).argmax(1)
            #print(pred)
            #print(gt)
            sum += (pred != (gt)).detach().sum().cpu()
    if model_name!=None:
        print(model_name + 'wrong rate = {:.2%}'.format(sum / 1000.0))
    else:
        print('wrong rate = {:.2%}'.format(sum / 1000.0))
def main():

    model_names = ['inceptionv3' ,
              'inceptionv4', #'densenet121', 'densenet169',
              'densenet201', #'densenet161', 'resnet18', 'resnet34',
              'resnet50', #'resnet101', 'resnet152', #'vgg11', 'vgg13','vgg16', 
              'vgg19', 
              'senet154',#'xception',
              'inceptionresnetv2',
              "pnasnet5large"
              ]

    for model_name in model_names:
        #verify(model_name)
        print("===================================================")
        
        
    net = torchvision.models.inception_v3(pretrained=True)
    print("v3")
    verify(None, net)
    
    net = torchvision.models.mobilenet_v2(pretrained=True)
    print("mobilenet_v2")
    verify(None, net)


    net = torchvision.models.resnext101_32x8d(pretrained=True)
    print("resnext101_32x8d")
    verify(None, net)

    net = torchvision.models.mnasnet1_0(pretrained=True)
    print("mnasnet1_0")
    verify(None, net)

    net=torchvision.models.wide_resnet101_2(pretrained=True)
    print("wide_resnet101_2")
    verify(None, net)
    

if __name__ == '__main__':
    main()