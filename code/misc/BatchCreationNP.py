import random

def RandSimilarityPerturbationNP(I1, HObj, PatchSize, ImageSize=None, Vis=False):
    if(ImageSize is None):
        ImageSize = np.array(np.shape(I1))

    H, Params = HObj.GetRandReducedHICSTN(TransformType = 'psuedosimilarity')
    # Warp using Numpy 
    opt = warp2.Options(PatchSize=ImageSize, MiniBatchSize=1, warpType= ['pseudosimilarity'])
    Params = np.squeeze(Params)
    H = np.squeeze(H)
    
    # Numpy based Warping
    I2 = warp2.transformImageNP(opt, I1[np.newaxis,:,:,:], H[np.newaxis,:,:])[0]
    
    # Crop in center for PatchSize
    P1 = iu.CenterCrop(I1, PatchSize)
    P2 = iu.CenterCrop(I2, PatchSize)
    
    if(Vis is True):
        cv2.imshow('I1, I2', np.hstack((I1, I2)))
        cv2.imshow('I1, I2', np.hstack((P1, P2)))
        cv2.waitKey(0)
    
    # P1 is I1 cropped to patch Size
    # P2 is I1 Crop Warped (I2 Crop)
    # H is Homography
    # Params is the stuff H is made from 
    return I1, I2, P1, P2, H, Params


def GenerateBatchNP(TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize):
    """
    Inputs: 
    DirNames - Full path to all image files without extension
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of I1 images after standardization and cropping/resizing to ImageSize
    HomeVecBatch - Batch of Homing Vector labels
    """
    IBatch = [] # P1, P2
    I1Batch = []
    I2Batch = []
    HBatch = []
    ParamsBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainNames)-1)
        RandImageName = BasePath + os.sep + TrainNames[RandIdx] 
        I = cv2.imread(RandImageName)
        I = iu.RandomCrop(I, OriginalImageSize)
        if (I is None):
            continue
        ImageNum += 1

        # Similarity and Patch generation 
        I1, I2, P1, P2, H, Params = RandSimilarityPerturbation(I, HObj, PatchSize, ImageSize = None, Vis = False)
        # [Scale, Yaw, Shear, T2D]
        # CompositionsExtracted = np.array([Compositions[0][0]-1, Compositions[3][0], Compositions[3][0]]) # Neglect the stuff you don't need
        
        ICombined = np.dstack((P1[:,:,0:3], P2[:,:,0:3]))
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IS = iu.StandardizeInputs(np.float32(ICombined))

        # Append All Images and Mask
        IBatch.append(IS)
        I1Batch.append(P1)
        I2Batch.append(P2)
        HBatch.append(H)
        ParamsBatch.append(Params)

    # CompositionsBatch = np.squeeze(CompositionsBatch)
    return IBatch, I1Batch, I2Batch, HBatch, ParamsBatch 
