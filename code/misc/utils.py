import cv2
import numpy as np
import importlib
from datetime import datetime
import getpass
import copy
import platform
from scipy.ndimage import gaussian_filter
import tensorflow as tf
clip_val = 50


import misc.MiscUtils as mu
import misc.ImageUtils as iu

def PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('GPU Used: {}'.format(Args.GPUDevice), 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                          VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate), 'yellow')
    cprint('Num Params: {}'.format(NumParams), 'green')
    cprint('Num FLOPs: {}'.format(NumFlops), 'green')
    cprint('Estimated Model Size (MB): {}'.format(ModelSize), 'green')
    cprint('Augmentations Used: {}'.format(Args.Augmentations), 'green')
    cprint('Model loaded from: {}'.format(Args.CheckPointPath), 'red')
    cprint('Images used for Testing are in: {}'.format(Args.BasePath), 'red')


def WriteHeader(PredOuts, Args, NumParams, NumFlops, ModelSize, VN):
    PredOuts.write('Network Used: {}\n'.format(Args.NetworkName))
    PredOuts.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                                      VN.NumBlocks, VN.NumSubBlocks,  VN.DropOutRate))
    PredOuts.write('Num Params: {}\n'.format(NumParams))
    PredOuts.write('Num FLOPs: {}\n'.format(NumFlops))
    PredOuts.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
    PredOuts.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
    PredOuts.write('Images used for Testing are in: {}\n'.format(Args.BasePath))



def AccumPreds(prVals):
    prValAccum = None
    prValsAccum = []
    for prVali in prVals:
        if prValAccum == None:
            prValAccum = prVali
            prValsAccum.append(prValAccum)
            continue
                
        prValAccum = tf.compat.v1.image.resize_bilinear(prValAccum, [prVali.shape[1], prVali.shape[2]])
        prValAccum += prVali
        prValsAccum.append(prValAccum)
    
    return prValAccum,prValsAccum



def setup_full_model0(InputPH, Args):
    # Create Network Object with required parameters
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(Args.Net, ClassName)
    VN = Network(InputPH = InputPH, InitNeurons = Args.InitNeurons, NumSubBlocks = Args.NumSubBlocks, Suffix = '', NumOut = Args.NumOut, ExpansionFactor = Args.ExpansionFactor, UncType = None)
    prVal0 = VN.Network()
    if Args.NetworkName == "Network.MultiScaleResNet" or Args.NetworkName == "Network.ResNetAniMscale" or Args.NetworkName == "Network.MultiScaleMBResNet":
        accumOut = AccumPreds(prVal0)
        prVal = accumOut[0][...,0:2]
        prValFull = accumOut[1]
    Saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    Saver.restore(sess, Args.checkpoint)
    # warped_img = tf_warp(InputPH[...,0:3], prVal, Args.PatchSize0, Args.PatchSize1)
    return VN, prVal, sess, prVal0, prValFull

def setup_full_model(InputPH, Args):
    # Create Network Object with required parameters
    Args.NetworkName = "Network.MultiScaleResNet"
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(Args.Net, ClassName)
    VN = Network(InputPH = InputPH, InitNeurons = 32, NumSubBlocks = 2, Suffix = '', NumOut = Args.NumOut, ExpansionFactor = 2, UncType = None)
    prVal = VN.Network()
    if Args.NetworkName == "Network.MultiScaleResNet" or Args.NetworkName == "Network.ResNetAniMscale" or Args.NetworkName == "Network.MultiScaleMBResNet":
        accumOut = AccumPreds(prVal)
        prVal = accumOut[0][...,0:2]
        prValFull = accumOut[1]
    Saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    Saver.restore(sess, Args.checkpoint)
    # warped_img = tf_warp(InputPH[...,0:3], prVal, Args.PatchSize0, Args.PatchSize1)
    return VN, prVal, sess

def is_multiscale(Args):
    return Args.NetworkName == "Network.MultiScaleResNet" or Args.NetworkName == "Network.ResNetAniMscale" or Args.NetworkName == "Network.MultiScaleMBResNet"

def get_sintel_batch(img1_filename, img2_filename, flo_filename, patch_size):
    gtBatch = []

    img1 = cv2.imread(img1_filename)           
    img2 = cv2.imread(img2_filename)
    if img1 is None or img2 is None:
        return None, None

    try:
        gt = mu.readFlow(flo_filename)
    except:
        return None, None

    gt = np.clip(gt, a_min=-clip_val, a_max=clip_val)
    
    try:
        I = np.concatenate((img1, img2, gt), axis=2)
    except Exception as e:
        print(e)
        return None, None

    I = iu.ResizeNearestCrop(I, patch_size)

    if I is None:
        return None, None

    if gt is None:
        return None, None

    P1 = I[:,:,:3]
    P2 = I[:,:,3:6]
    gt = I[:,:,6:8]

    img_comb = np.concatenate((P1, P2), axis=2, dtype=np.float32)

    return img_comb, [gt]

def read_sintel_list(args):
    """
    https://github.com/hellochick/PWCNet-tf2/blob/674b593f2d3c02b89670edad900f645a4b1c7ce4/data_loader.py#L19
    """
    fp = open(args.data_list, 'r')
    line = fp.readline()

    im1_filenames, im2_filenames, flo_filenames = [], [], []
    while line:
        im1_fn, im2_fn, flo_fn = line.replace('\n', '').split(' ')
        
        im1_filenames.append(im1_fn)
        im2_filenames.append(im2_fn)
        flo_filenames.append(flo_fn)
        
        line = fp.readline()
    fp.close()

    return im1_filenames, im2_filenames, flo_filenames
