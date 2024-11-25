
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import cv2
import sys
import os
import glob
import re
import misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import misc.TFUtils as tu
from misc.DataHandling import *
from misc.BatchCreationTF import *
from misc.Decorators import *
from misc.FlowVisUtilsTF import *
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform
import pdb
import misc.FlowPolar as fp
import misc.FlowShift as fs

from misc.Losses import AssemblePreds, AssembleImages

def AccumPreds(prVals):
    prValAccum = None
    for prVali in prVals:
        if prValAccum == None:
            prValAccum = prVali
            continue
                
        prValAccum = tf.compat.v1.image.resize(prValAccum, [prVali.shape[1], prVali.shape[2]])
        prValAccum += prVali
    
    return prValAccum

def TensorBoard(loss, I1PH, I2PH, prVal, Label1PH, Label2PH, Args):
    # Create a summary to monitor loss tensor    
    tf.compat.v1.summary.scalar('LossEveryIter', loss)


    tf.compat.v1.summary.image('I1Patch', I1PH[:,:,:,0:3], max_outputs=1)
    # tf.compat.v1.summary.image('I2Patch', I2PH[:,:,:,0:3], max_outputs=3)
    # tf.summary.image('prVal', prVal[:,:,:,0:1], max_outputs=3)
    # tf.summary.image('Label1', Label1PH[:,:,:,0:1], max_outputs=3)
    # tf.summary.image('Label2', Label2PH[:,:,:,0:1], max_outputs=3)
    
    if Args.NetworkName == "Network.MultiScaleResNet":
        Label1FlowVis, _, _ = flow_viz_tf(Label1PH)
        Label2FlowVis, _, _ = flow_viz_tf(Label2PH)
        prFlowVis_final, _, _ = flow_viz_tf(AccumPreds(prVal)) ## TODO how does this actually work? 
        prFlowVis_prefinal, _, _ = flow_viz_tf(prVal[-2][:,:,:,0:2])
    elif Args.NetworkName == "Network.MultiScaleMBResNet":
        Label1FlowVis, _, _ = flow_viz_tf(Label1PH)
        Label2FlowVis, _, _ = flow_viz_tf(Label2PH)
        prFlowVis_final, _, _ = flow_viz_tf(AccumPreds(prVal)[:,:,:,0:2]) 
        prFlowVis_prefinal, _, _ = flow_viz_tf(prVal[-2][:,:,:,0:2])
        prMotionBoundaryVis = AccumPreds(prVal)[:,:,:,4]
    else:
        Label1FlowVis, _, _ = flow_viz_tf(fs.unshift_flow_tf(Label1PH))
        Label2FlowVis, _, _ = flow_viz_tf(fs.unshift_flow_tf(Label2PH))
        prFlowVis, _, _ = flow_viz_tf(fs.unshift_flow_tf(prVal[:,:,:,0:2]))
    
    
    
    if Args.NetworkName == "Network.MultiScaleResNet":
        tf.compat.v1.summary.image('prVal_final', prFlowVis_final[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('prVal_prefinal', prFlowVis_prefinal[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('Label1', Label1FlowVis[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('Label2', Label2FlowVis[:,:,:,0:3], max_outputs=1)
    elif Args.NetworkName == "Network.MultiScaleMBResNet":
        tf.compat.v1.summary.image('prVal_final', prFlowVis_final[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('prVal_prefinal', prFlowVis_prefinal[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('prVal_MB', prMotionBoundaryVis[...,None], max_outputs=1)
        tf.compat.v1.summary.image('Label1', Label1FlowVis[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('Label2', Label2FlowVis[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('LabelMB01', LabelMB01Vis, max_outputs=1)
    else:
        tf.compat.v1.summary.image('Label1', Label1FlowVis[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('Label2', Label2FlowVis[:,:,:,0:3], max_outputs=1)
        tf.compat.v1.summary.image('prVal', prFlowVis[:,:,:,0:3], max_outputs=1)

    tf.compat.v1.summary.histogram('Label1Hist', Label1PH)
    tf.compat.v1.summary.histogram('Label2Hist', Label2PH)
    if(Args.UncType == 'Aleatoric'):
        eps = 1e-6  # To avoid Inf
        MaxVal = 10.
        prLogvarX = tf.clip_by_value(prVal[:,:,:,2:3], eps, MaxVal)
        prLogvarY = tf.clip_by_value(prVal[:,:,:,3:4], eps, MaxVal)
        prLogvar = tf.clip_by_value(prVal[:,:,:,2:4], eps, MaxVal)
        tf.compat.v1.summary.histogram('prLogValHist', prLogvar)
        tf.compat.v1.summary.histogram('prValHist', prVal[:,:,:,0:2])
        tf.compat.v1.summary.image('AleatoricUncX', tf.exp(prLogvarX), max_outputs=1)
        tf.compat.v1.summary.image('AleatoricUncY', tf.exp(prLogvarY), max_outputs=1)
        tf.compat.v1.summary.histogram('AleatoricUncHistX', tf.exp(prLogvarX))
        tf.compat.v1.summary.histogram('AleatoricUncHistY', tf.exp(prLogvarY))
    elif(Args.UncType == 'Inlier'):
        tf.compat.v1.summary.histogram('FlowPred', prVal[:,:,:,0:2])
        tf.compat.v1.summary.image('Inlier', prVal[:,:,:,2:3], max_outputs=1)
        tf.compat.v1.summary.histogram('Inlier', prVal[:,:,:,2:3])
    elif(Args.UncType == 'LinearSoftplus'):
        Eps = 1e-3
        MaxVal = 1e9
        # Sigmoid
        # tf.compat.v1.summary.image('ScaleX', tf.clip_by_value(1/tf.math.sigmoid(prVal[:,:,:,2:3] + Eps), -MaxVal, MaxVal), max_outputs=1)
        # tf.compat.v1.summary.image('ScaleY', tf.clip_by_value(1/tf.math.sigmoid(prVal[:,:,:,3:4] + Eps), -MaxVal, MaxVal), max_outputs=1)
        # tf.compat.v1.summary.histogram('Scale', tf.clip_by_value(1/tf.math.sigmoid(prVal + Eps), -MaxVal, MaxVal))
        # Softplus
        if (Args.NetworkName == 'Network.MultiScaleResNet' or Args.NetworkName == "Network.MultiScaleMBResNet"):
            prVal = AccumPreds(prVal)
        tf.compat.v1.summary.image('ScaleX', tf.clip_by_value(1/tf.math.softplus(prVal[:,:,:,2:3] + Eps), -MaxVal, MaxVal), max_outputs=1)
        tf.compat.v1.summary.image('ScaleY', tf.clip_by_value(1/tf.math.softplus(prVal[:,:,:,3:4] + Eps), -MaxVal, MaxVal), max_outputs=1)
        tf.compat.v1.summary.histogram('Scale', tf.clip_by_value(1/tf.math.softplus(prVal + Eps), -MaxVal, MaxVal))
        tf.compat.v1.summary.histogram('prValHist', prVal[:,:,:,0:2])
    else:
        if Args.NetworkName == "Network.MultiScaleResNet" or (Args.NetworkName == "Network.MultiScaleMBResNet"):
            tf.compat.v1.summary.histogram('prValHist', prVal[-1])
        else:
            tf.compat.v1.summary.histogram('prValHist', prVal)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.compat.v1.summary.merge_all()
    return MergedSummaryOP


def PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=False):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('Uncertainity Type: {}'.format(Args.UncType), 'yellow')
    cprint('GPU Used: {}'.format(Args.GPUDevice), 'yellow')
    cprint('Learning Rate: {}'.format(Args.LR), 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                          VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate), 'yellow')
    cprint('Num Params: {}'.format(NumParams), 'green')
    cprint('Num FLOPs: {}'.format(NumFlops), 'green')
    cprint('Estimated Model Size (MB): {}'.format(ModelSize), 'green')
    cprint('Loss Function used: {}'.format(Args.LossFuncName), 'green')
    cprint('Loss Function Weights: {}'.format(Args.Lambda), 'green')
    cprint('Reg Function used: {}'.format(Args.RegFuncName), 'green')
    cprint('Augmentations Used: {}'.format(Args.Augmentations), 'green')
    cprint('CheckPoints are saved in: {}'.format(Args.CheckPointPath), 'red')
    cprint('Logs are saved in: {}'.format(Args.LogsPath), 'red')
    cprint('Images used for Training are in: {}'.format(Args.BasePath), 'red')
    if(OverideKbInput):
        Key = 'y'
    else:
        PythonVer = platform.python_version().split('.')[0]
        # Parse Python Version to handle super accordingly
        if (PythonVer == '2'):
            Key = raw_input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
        else:
            Key = input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
    if(Key.lower() == 'y' or Key.lower() == 'yes'):
        # FileName = 'RunCommand.md'
        # with open(FileName, 'a+') as RunCommand:
        #     RunCommand.write('\n\n')
        #     RunCommand.write('{}\n'.format(datetime.now()))
        #     RunCommand.write('Username: {}\n'.format(Username))
        #     RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
        #     RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
        #     RunCommand.write('Uncertainity Type: {}\n'.format(Args.UncType))
        #     RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
        #     RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
        #                                                                                                                       VN.NumBlocks, VN.NumSubBlocks,  VN.DropOutRate))
        #     RunCommand.write('Num Params: {}\n'.format(NumParams))
        #     RunCommand.write('Num FLOPs: {}\n'.format(NumFlops))
        #     RunCommand.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
        #     RunCommand.write('Loss Function used: {}\n'.format(Args.LossFuncName))
        #     RunCommand.write('Loss Function Weights: {}\n'.format(Args.Lambda))
        #     RunCommand.write('Reg Function used: {}\n'.format(Args.RegFuncName))
        #     RunCommand.write('Augmentations Used: {}\n'.format(Args.Augmentations))
        #     RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
        #     RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
        #     RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        # cprint('Log written in {}'.format(FileName), 'yellow')
        FileName = Args.CheckPointPath + 'RunCommand.md'
        with open(FileName, 'w+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('Uncertainity Type: {}\n'.format(Args.UncType))
            RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                                              VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate))
            RunCommand.write('Num Params: {}\n'.format(NumParams))
            RunCommand.write('Num FLOPs: {}\n'.format(NumFlops))
            RunCommand.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
            RunCommand.write('Loss Function used: {}\n'.format(Args.LossFuncName))
            RunCommand.write('Loss Function Weights: {}\n'.format(Args.Lambda))
            RunCommand.write('Reg Function used: {}\n'.format(Args.RegFuncName))
            RunCommand.write('Augmentations Used: {}\n'.format(Args.Augmentations))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
    else:
        cprint('Log writing skipped', 'yellow')
        
    