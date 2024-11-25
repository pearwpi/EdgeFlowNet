import random
import os
import cv2
import numpy as np
import tensorflow as tf
import misc.ImageUtils as iu
import misc.FlowVisUtilsNP as fvu
import misc.MiscUtils as mu
import scipy.io as sio
import imageio 
import misc.FlowShift as fs

class BatchGeneration():
    def __init__(self):
        pass

    def ClipFlowValues(self, flow):
        mean = 0
        clip_value = 50
        min = -50
        ptp = 100
        flow_clipped = np.clip(flow, a_min=-clip_value, a_max=clip_value)
        #flow_shiftted = 255*(flow_clipped - min)/ptp
        return flow_clipped

    def GenerateBatchTFFT3D(self, Args):

        P1Batch = []
        P2Batch = []
        Label1Batch = []
        Label2Batch = []
        LabelMB01Batch = []
        
        ImageNum = 0
        while ImageNum < Args.MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(Args.TrainNames)-1)
            RandImageName1 = os.sep + Args.TrainNames[RandIdx]

            RandImageName1 = Args.BasePath + os.sep + Args.TrainNames[RandIdx]
            # Increment Image Number by 1
            ImgName = RandImageName1.rsplit('/', 1)[-1]
            ImgName = ImgName.split('.png')[0]

            RandImageName2 = RandImageName1.rsplit('/', 1)[0] + '/%04d'%(int(ImgName)+1) + '.png'
            
            LabelName1 = RandImageName1.replace(Args.BasePath, Args.LabelBasePath)
            LabelName1 = LabelName1.replace('frames_cleanpass', 'optical_flow')
            LabelName1 = LabelName1.replace('left', 'into_future/left')
            LabelName1 = LabelName1.replace(RandImageName1.rsplit('/', 1)[-1], 'OpticalFlowIntoFuture_' + RandImageName1.rsplit('/', 1)[-1])
            LabelName1 = LabelName1.replace('.png', '_L.pfm')
            
            LabelName2 = LabelName1.replace('OpticalFlowIntoFuture_', 'OpticalFlowIntoPast_')
            LabelName2 = LabelName2.replace('into_future', 'into_past')
            if not os.path.exists(RandImageName1) and not os.path.exists(RandImageName2) and not os.path.exists(LabelName1) and not os.path.exists(LabelName2):
                print(f" RandImageName1: {RandImageName1} ")
                continue
            # print(f"ImgName: {ImgName}")
            # print(f"RandImgName2: {RandImageName2}")
            # print(f"LabelName1: {LabelName1}")
            # print(f"LabelName2: {LabelName2}")

            I1 = cv2.imread(RandImageName1)           
            I2 = cv2.imread(RandImageName2)
            if(I1 is None or I2 is None):
                continue
            
            try:
                Label1 = mu.readFlow(LabelName1)
                Label2 = mu.readFlow(LabelName2)
                Label1 = np.divide(Label1, 12.5)
                Label1 = np.divide(Label2, 12.5)   
                LabelMB01 = Label2[:,:,0:1].astype(np.float32)
            except:
                print("I am here4")
                continue


            Label1 = self.ClipFlowValues(Label1)
            # Label1 = fvu.flow_polar(Label1[:,:,0], Label1[:,:,1])
            
            Label2 = self.ClipFlowValues(Label2)
            # Label2 = fvu.flow_polar(Label2[:,:,0], Label2[:,:,1])
            
            if Args.NetworkName != "Network.MultiScaleResNet" \
                and Args.NetworkName != "Network.MultiScaleMBResNet": 
                Label1 = fs.shift_flow_tf(Label1)
                Label2 = fs.shift_flow_tf(Label1)
        
            try:
                I = np.concatenate((I1, I2, Label1, Label2, LabelMB01), axis=2)
                IOrg = np.concatenate((I1, I2), axis=2)
            except Exception as e:
                print(e)
                print("I am here")
                continue
            
            I = iu.RandomCrop(I, Args.PatchSize)

            if (I is None):
                print("I am here2")
                continue

            P1 = I[:,:,:3]
            P2 = I[:,:,3:6]
            Label1 = I[:,:,6:8]
            Label2 = I[:,:,8:10]
            LabelMB01 = I[:,:,10:]
                
            
            ImageNum += 1
            P1Batch.append(P1)
            P2Batch.append(P2)
            Label1Batch.append(Label1)
            Label2Batch.append(Label2)
            LabelMB01Batch.append(LabelMB01)

            
        ICombined = np.concatenate((P1Batch, P2Batch), axis=3)
        
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IBatch = np.float32(ICombined)
        
        return IBatch, P1Batch, P2Batch, Label1Batch, Label2Batch, LabelMB01Batch

    def GenerateBatchTF(self, Args):
        if (Args.Dataset == "FT3D"):
            return self.GenerateBatchTFFT3D(Args)
        P1Batch = []
        P2Batch = []
        Label1Batch = []
        Label2Batch = []
        
        ImageNum = 0
        while ImageNum < Args.MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(Args.TrainNames)-1)
            RandImageName1 = os.sep + Args.TrainNames[RandIdx]

            if(Args.Dataset == 'FC2'):
                RandImageName2 = RandImageName1.replace('img_0', 'img_1')
                
                LabelName1 = RandImageName1.replace('img_0.png', 'flow_01.flo')
                LabelName2 = RandImageName1.replace('img_0.png', 'flow_10.flo')

            I1 = cv2.imread(RandImageName1)           
            I2 = cv2.imread(RandImageName2)
            if(I1 is None or I2 is None):
                continue
            
            try:
                Label1 = mu.readFlow(LabelName1)
                Label2 = mu.readFlow(LabelName2)
            except:
                print("I am here4")
                continue

            Label1 = self.ClipFlowValues(Label1)
            # Label1 = fvu.flow_polar(Label1[:,:,0], Label1[:,:,1])
            
            Label2 = self.ClipFlowValues(Label2)
            # Label2 = fvu.flow_polar(Label2[:,:,0], Label2[:,:,1])
            
            if Args.NetworkName != "Network.MultiScaleResNet" \
                or Args.NetworkName != "Network.MultiScaleMBResNet": 
                Label1 = fs.shift_flow_tf(Label1)
                Label2 = fs.shift_flow_tf(Label1)

        
            try:
                I = np.concatenate((I1, I2, Label1, Label2), axis=2)
                IOrg = np.concatenate((I1, I2), axis=2)
            except Exception as e:
                print(e)
                print("I am here")
                continue
            
            I = iu.RandomCrop(I, Args.PatchSize)

            if (I is None):
                print("I am here2")
                continue

            P1 = I[:,:,:3]
            P2 = I[:,:,3:6]
            Label1 = I[:,:,6:8]
            Label2 = I[:,:,8:10]
            LabelMB01 = I[:,:,10:]
                
            
            ImageNum += 1
            P1Batch.append(P1)
            P2Batch.append(P2)
            Label1Batch.append(Label1)
            Label2Batch.append(Label2)
            
        ICombined = np.concatenate((P1Batch, P2Batch), axis=3)
        
        IBatch = np.float32(ICombined)
        
        return IBatch, P1Batch, P2Batch, Label1Batch, Label2Batch
