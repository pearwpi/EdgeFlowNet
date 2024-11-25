#!/usr/bin/env python3
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
import importlib
from datetime import datetime
import misc.FlowPolar as fp
import misc.FlowShift as fs
from misc.Losses import Loss
from misc.TensorBoardDisplay import TensorBoard,PrettyPrint

# Don't generate pyc codes
sys.dont_write_bytecode = True

# Disable Eager Execution to use TF1 code in TF2
tf.compat.v1.disable_eager_execution()


@Scope
def Optimizer(OptimizerParams, loss):
    Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3])
    Gradients = Optimizer.compute_gradients(loss)
    #Gradients = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in Gradients]
    OptimizerUpdate = Optimizer.apply_gradients(Gradients)
    # Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
    # Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)
    return OptimizerUpdate

def TrainOperation(InputPH, I1PH, I2PH, Label1PH, Label2PH, args):
    # Create Network Object with required parameters
    args.NetworkName = "Network.MultiScaleResNet"
    ClassName = args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(args.Net, ClassName)
    VN = Network(InputPH = InputPH, 
                InitNeurons = 32, 
                NumSubBlocks = 2, 
                Suffix = '', 
                NumOut = args.NumOut, 
                UncType = "LinearSoftplus", 
                ExpansionFactor=2)
    prVal = VN.Network()
    loss = Loss(I1PH, I2PH, Label1PH, Label2PH, prVal, args)
    OptimizerUpdate = Optimizer(args.OptimizerParams, loss)
    MergedSummaryOP = TensorBoard(loss, I1PH, I2PH, prVal, Label1PH, Label2PH, args)
    Saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:       
        if args.LatestFile is not None:
            Saver.restore(sess, args.CheckPointPath + args.LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in args.LatestFile.split('a')[0] if c.isdigit())) + 1
            print('Loaded latest checkpoint with the name ' + args.LatestFile + '....')
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        bg = BatchGeneration()

        args.Augmentations = 'None'

        NumParams = tu.FindNumParams(1)
        NumFlops = tu.FindNumFlops(sess, 1)
        ModelSize = tu.CalculateModelSize(1)

        PrettyPrint(args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=True)
        Writer = tf.compat.v1.summary.FileWriter(args.LogsPath, graph=tf.compat.v1.get_default_graph())

        if args.SaveTestModel:
            SaveName =  args.CheckPointPath + str(0) + 'a' + str(0) + 'model.ckpt'
            Saver.save(sess,  save_path=SaveName)
            print(SaveName + ' Model Saved...')
            exit(0)

        for Epochs in tqdm(range(StartEpoch, args.NumEpochs)):
            NumIterationsPerEpoch = int(args.NumTrainSamples/args.MiniBatchSize/args.DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                IBatch, P1Batch, P2Batch, Label1Batch, Label2Batch = bg.GenerateBatchTF(args)
                
                FeedDict = {VN.InputPH: IBatch, 
                            I1PH: P1Batch, 
                            I2PH: P2Batch, 
                            Label1PH: Label1Batch, 
                            Label2PH: Label2Batch}
                _, _, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                Writer.flush()

                if PerEpochCounter % args.SaveCheckPoint == 0:
                    SaveName =  args.CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print(SaveName + ' Model Saved...')

            SaveName = args.CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print(SaveName + ' Model Saved...')

    PrettyPrint(args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=True)

        
        
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--BasePath', default='/Datasets/FlyingChairs2/', help='Base path of images')
    parser.add_argument('--NumEpochs', type=int, default=400, help='Number of Epochs to Train for')
    parser.add_argument('--SaveTestModel', action='store_true', help='Number of Epochs to Train for')
    parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch')
    parser.add_argument('--MiniBatchSize', type=int, default=16, help='Size of the MiniBatch to use')
    parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointPath?')
    parser.add_argument('--ExpansionFactor', type=float, default=2, help='PathSizeChannels')
    parser.add_argument('--RemoveLogs', type=int, default=0, help='Delete log Files from ./Logs?')
    parser.add_argument('--LossFuncName', default='MultiscaleSL1-1', help='Choice of Loss functions, choose from SL2, PhotoL1, PhotoChab, PhotoRobust')
    parser.add_argument('--RegFuncName', default='None', help='Choice of regularization function, choose from None, C (Cornerness).')
    parser.add_argument('--NumSubBlocks', type=int, default=3, help='Number of sub blocks')
    parser.add_argument('--ExperimentFileName', default="default", help='Default experiement name')
    parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU')
    parser.add_argument('--DataAug', type=int, default=0, help='Do you want to do Data augmentation?')
    parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--Suffix', default='', help='Suffix for Naming Network ')
    parser.add_argument('--Dataset', default='FC2', help='Dataset: FC2 for Flying Chairs 2 or FT3D for Flying Things 3D')
    parser.add_argument('--data_list', default='code/dataset_paths/MPI_Sintel_Final_train_list.txt', help='list of data')
    args = parser.parse_args()

    
    tu.SetGPU(args.GPUDevice)

    args.Net = importlib.import_module('network.MultiScaleResNet')


    if(args.RemoveLogs != 0):
        shutil.rmtree(os.getcwd() + os.sep + 'Logs' + os.sep)

    args = SetupAll(args)

        
    InputPH = tf.compat.v1.placeholder(tf.float32, shape=(args.MiniBatchSize, args.PatchSize[0], args.PatchSize[1], 2*args.PatchSize[2]), name='Input')

    # PH for losses
    I1PH = tf.compat.v1.placeholder(tf.float32, shape=(args.MiniBatchSize, args.PatchSize[0], args.PatchSize[1], args.PatchSize[2]), name='I1')
    I2PH = tf.compat.v1.placeholder(tf.float32, shape=(args.MiniBatchSize, args.PatchSize[0], args.PatchSize[1], args.PatchSize[2]), name='I2')
    
    Label1PH =  tf.compat.v1.placeholder(tf.float32, shape=(args.MiniBatchSize, args.PatchSize[0], args.PatchSize[1], args.NumOut), name='Label1')
    Label2PH =  tf.compat.v1.placeholder(tf.float32, shape=(args.MiniBatchSize, args.PatchSize[0], args.PatchSize[1], args.NumOut), name='Label2')
        
    TrainOperation(InputPH, I1PH, I2PH, Label1PH, Label2PH,args)
    
    
if __name__ == '__main__':
    main()

