import numpy as np
import tensorflow as tf
from misc.warp import tf_warp

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
    if Args.NetworkName == "Network.MultiScaleResNet" or Args.NetworkName == "Network.ResNetAniketMscale" or Args.NetworkName == "Network.MultiScaleMBResNet":
        accumOut = AccumPreds(prVal0)
        prVal = accumOut[0][...,0:2]
        prValFull = accumOut[1]
    Saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    Saver.restore(sess, Args.CheckPointPath)
    # warped_img = tf_warp(InputPH[...,0:3], prVal, Args.PatchSize0, Args.PatchSize1)
    return VN, prVal, sess, prVal0, prValFull

def setup_full_model(InputPH, Args):
    # Create Network Object with required parameters
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(Args.Net, ClassName)
    VN = Network(InputPH = InputPH, InitNeurons = Args.InitNeurons, NumSubBlocks = Args.NumSubBlocks, Suffix = '', NumOut = Args.NumOut, ExpansionFactor = Args.ExpansionFactor, UncType = None)
    prVal = VN.Network()
    if Args.NetworkName == "Network.MultiScaleResNet" or Args.NetworkName == "Network.ResNetAniketMscale" or Args.NetworkName == "Network.MultiScaleMBResNet":
        accumOut = AccumPreds(prVal)
        prVal = accumOut[0][...,0:2]
        prValFull = accumOut[1]
    Saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    Saver.restore(sess, Args.CheckPointPath)
    # warped_img = tf_warp(InputPH[...,0:3], prVal, Args.PatchSize0, Args.PatchSize1)
    return VN, prVal, sess


def is_multiscale(Args):
    return Args.NetworkName == "Network.MultiScaleResNet" or Args.NetworkName == "Network.ResNetAniketMscale" or Args.NetworkName == "Network.MultiScaleMBResNet"
