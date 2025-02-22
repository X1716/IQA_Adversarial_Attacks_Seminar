import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter

MEAN = torch.Tensor([0.485, 0.456, 0.406])
STD = torch.Tensor([0.229, 0.224, 0.225])
# вспомогательные функции
def normalize(im):
    '''
    Normalize a given tensor using Imagenet statistics

    '''
    mean = MEAN.cuda() if im.is_cuda else MEAN
    std = STD.cuda() if im.is_cuda else STD

    if im.dim() == 4:
        im = im.transpose(1, 3)
        im = (im - mean) / std
        im = im.transpose(1, 3)
    else:
        im = im.transpose(0, 2)
        im = (im - mean) / std
        im = im.transpose(0, 2)

    return im


def forward(model, classifier, opt, data):
    out_class, out_reg = model(data['L'], data['hints'].clamp(-1,1), data['mask'].clamp(-.5,.5))
    out_rgb = lab2rgb(torch.cat((data['L'], out_reg), dim=1), opt).clamp(0,1)
    y_pred = classifier(normalize(out_rgb))
    return out_rgb, y_pred

def compute_loss(opt, y, criterion):
    t = torch.LongTensor([opt.target]*y.shape[0]).cuda()
    loss = criterion(y, t)
    if not opt.targeted:
        loss *= -1
    return loss

def compute_class(opt, y, num_labels=5):
    y_softmax = F.softmax(y)
    # assume we are using batch of 1 here and no jpg loss
    val, idx = y_softmax[0].sort(descending=True)
    labels = [(opt.idx2label[idx[i]], round(val[i].item(), 3)) for i in range(num_labels)]

    return val, idx, labels

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    return out

def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-50.)/100.
    ab_rs = lab[:,1:,:,:]/110.
    out = torch.cat((l_rs,ab_rs),dim=1)
    return out

def lab2rgb(lab_rs, opt=None):
    #l = lab_rs[:,[0],:,:]*opt.l_norm + opt.l_cent
    l = lab_rs[:,[0],:,:]*100. + 50.
    ab = lab_rs[:,1:,:,:]*110.
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out

