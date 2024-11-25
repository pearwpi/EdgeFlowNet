#!/usr/bin/env python3
import tensorflow as tf
import cv2
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import glob
import re
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from misc.MiscUtils import *
from misc.FlowVisUtilsNP import *
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import misc.FlowVisUtilsNP as fvu
import misc.SintelFlowViz as svu
import math as m
from tqdm import tqdm
import misc.TFUtils as tu
from misc.Decorators import *
import misc.FlowShift as fs
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform
import misc.TFUtils as tu
from scipy.ndimage import gaussian_filter
import misc.FlowPolar as fp
from misc.processor import FlowPostProcessor

# Don't generate pyc codes
sys.dont_write_bytecode = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from misc.utils import *
# Disable Eager Execution to use TF1 code in TF2
tf.compat.v1.disable_eager_execution()


def TestOperation(InputPH, Args):
    VN, prVal, sess = setup_full_model(InputPH, Args)

    processor = FlowPostProcessor("full", is_multiscale(Args))
    
    im1_filenames, im2_filenames, flo_filenames = read_sintel_list(Args)

    IBatch = np.random.rand(Args.InputPatchSize[0], Args.InputPatchSize[1], 2*Args.InputPatchSize[2])
    Label1Batch = np.random.rand(1,Args.InputPatchSize[0], Args.InputPatchSize[1], Args.NumOut)

    for i in tqdm(range(0,len(im1_filenames))):
        IBatch, Label1Batch = get_sintel_batch(im1_filenames[i], im2_filenames[i], flo_filenames[i], Args.PatchSize)
        if(IBatch is None):
            continue
    
        FeedDict = {VN.InputPH: IBatch[None,...]}
        
        prediction_full = sess.run([prVal], feed_dict=FeedDict)[0]
    
        processor.update(Label1Batch, prediction_full, Args)


    processor.print()

def main():
    Parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument('--checkpoint', default='/root/optical/models/brown/', help='Path to save checkpoints')
    Parser.add_argument('--gpu_device', type=int, default=0, help='What GPU do you want to use? -1 for CPU')
    Parser.add_argument('--patch_dim_0', default=416, type=int, help='patch size 0')
    Parser.add_argument('--patch_dim_1', default=1024, type=int, help='patch size 1')
    Parser.add_argument('--patch_channels', default=3, type=int, help='patch size channels')
    Parser.add_argument('--patch_delta', default=20, type=int, help='additional patch delta')
    Parser.add_argument('--uncertainity', action='store_true', help='is uncertainity')
    Parser.add_argument('--data_list', default='./Misc/MPI_Sintel_train_clean.txt', help='list of sintel data')
    args = Parser.parse_args()

    tu.SetGPU(args.gpu_device)

    args.Net = importlib.import_module('network.MultiScaleResNet')

    args.InputPatchSize = np.array([args.patch_dim_0, args.patch_dim_1, args.patch_channels])
    args.PatchSize = args.InputPatchSize
    args.NumOut = 2

    if args.uncertainity:
        args.NumOut = 4

    InputPH = tf.compat.v1.placeholder(tf.float32, shape=(1, args.InputPatchSize[0], args.InputPatchSize[1], 2*args.InputPatchSize[2]), name='Input')
    
    TestOperation(InputPH, args)

if __name__ == '__main__':
    main()
