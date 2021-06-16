import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from skimage.restoration import (denoise_wavelet, estimate_sigma)

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def estimatePSF(img, psf_size = [5,5]):
    #img is 2D
    G = np.log(abs(np.fft.fft2(img))) #Fourier transform
#     print(G.shape)
    deltaG = G - medfilt2d(G)  
    lambd = 0.05 * abs(deltaG) #Threshold
    R = np.sign(deltaG)*np.maximum(0,abs(deltaG)-lambd)
    GR = G - R
    
    #wavelet denosing
#     sigma_est = estimate_sigma(GR) 
#     https://scikit-image.org/docs/dev/api/skimage.restoration.html#r3b8ec6d23a4e-2
#     BayesShrink: I think it's similar to level dependent threshold
    im_visushrink = denoise_wavelet(GR, method = 'BayesShrink',mode = 'soft', wavelet = 'sym9',
                                    wavelet_levels = 4, rescale_sigma=True)
    H = np.exp(im_visushrink)
    psf = abs(otf2psf(H, psf_size))
    return H,psf
    
#xudong ma 19/05/2021
#https://vimsky.com/examples/detail/python-method-torch.fft.html
#https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py 
def otf2psf(otf, outsize=None):
    insize = np.array(otf.shape)
    psf = np.fft.ifftn(otf, axes=(0, 1))
    for axis, axis_size in enumerate(insize):
        psf = np.roll(psf, np.floor(axis_size / 2).astype(int), axis=axis)
    if type(outsize) != type(None):
        insize = np.array(otf.shape)
        outsize = np.array(outsize)
        n = max(np.size(outsize), np.size(insize))
        # outsize = postpad(outsize(:), n, 1);
        # insize = postpad(insize(:) , n, 1);
        colvec_out = outsize.flatten().reshape((np.size(outsize), 1))
        colvec_in = insize.flatten().reshape((np.size(insize), 1))
        outsize = np.pad(colvec_out, ((0, max(0, n - np.size(colvec_out))), (0, 0)), mode="constant")
        insize = np.pad(colvec_in, ((0, max(0, n - np.size(colvec_in))), (0, 0)), mode="constant")

        pad = (insize - outsize) / 2
        if np.any(pad < 0):
            print("otf2psf error: OUTSIZE must be smaller than or equal than OTF size")
        prepad = np.floor(pad)
        postpad = np.ceil(pad)
        dims_start = prepad.astype(int)
        dims_end = (insize - postpad).astype(int)
        for i in range(len(dims_start.shape)):
            psf = np.take(psf, range(dims_start[i][0], dims_end[i][0]), axis=i)
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    return psf
    

def sensor_gain(us_img, mr_img, mbSize = 1):
    
    #mbSize is the width of steps
    #the size of the window should be (mbSize+1)*(mbSize+1)
    us_np = np.array(us_img)
    mr_np = np.array(mr_img)
    row = us_np.shape[0]
    col = us_np.shape[1]
    us_beta = np.zeros(us_np.shape)
    mr_beta = np.zeros(mr_np.shape)
    #The end value is not right for any mbSize, however, it's right when mbSize=1
    for i in range(0,row-mbSize,mbSize+1):
        for j in range(0,col-mbSize,mbSize+1):
            R_us = us_np[i:i+mbSize+1,j:j+mbSize+1]
            R_mr = mr_np[i:i+mbSize+1,j:j+mbSize+1]
            
            f_us = R_us.reshape(-1,1,order='F') #reshape cloumn first
            f_mr = R_mr.reshape(-1,1,order='F')
            
            vK = np.hstack((f_us,f_mr))
            variance = np.zeros((2,2))
            for k in range(0,vK.shape[0]):
                temp_variance = vK[k][None].T
                variance = variance+temp_variance.dot(temp_variance.T)
            # return eigenvalue (array) and normalized eigenvectors (length=1) ,column corresponding
            eigenvalue, eigenvector = np.linalg.eig(variance)
            principal_idx = abs(eigenvalue).argmax()
            sensor_gain = abs(eigenvector[:,principal_idx]) #get the principal eigenvector (abs)
            local_us_beta = sensor_gain[0]
            local_mr_beta = sensor_gain[1]
            if local_us_beta == local_mr_beta:
                local_us_beta = 1
                local_mr_beta = 1
            
            us_beta[i:i+mbSize+1,j:j+mbSize+1] = local_us_beta * np.ones((mbSize+1,mbSize+1))
            mr_beta[i:i+mbSize+1,j:j+mbSize+1] = local_mr_beta * np.ones((mbSize+1,mbSize+1))
                                                                     
#     normalize
    norm = np.square(us_beta) + np.square(mr_beta)
    us_result = us_beta/np.sqrt(norm)
    mr_result = mr_beta/np.sqrt(norm) 
    us_result[np.isnan(us_result)]=1
    mr_result[np.isnan(mr_result)]=1
    return us_result, mr_result   

def test():
    print("LLLLLL")


def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
#     img = Image.open(path)
    img = Image.open(path).convert('RGB') #Xudong adding
    return img

def samesize_images(us_path, mr_path):
    us_img = Image.open(us_path)
    mr_img = Image.open(mr_path)
    us_size = us_img.size
    mr_size = mr_img.size
    if us_size[0] == mr_size[0]:
        return us_path, mr_path
    elif us_size[0] > mr_size[0]: #magnify
        mr_img = mr_img.resize(us_size,Image.BICUBIC)
    else: #shrink
        mr_img = mr_img.resize(us_size,Image.ANTIALIAS)
    mr_img.save("data/denoising/mri_mod_samesize.png")
    mr_path = "data/denoising/mri_mod_samesize.png"
    return us_path, mr_path 


def get_image(path, imsize=-1):
    """Load an image and resize to a specific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC) #magnify
        else:
            img = img.resize(imsize, Image.ANTIALIAS) #shrink

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

#xudong add a new axis and values divid 255
def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1] 
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...] #add a new axis

    return ar.astype(np.float32) / 255.
#     return ar.astype(np.folat32)

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
#     ar = img_np.astype(np.uint16)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False
