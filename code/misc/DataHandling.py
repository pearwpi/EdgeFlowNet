import misc.ImageUtils as iu
import os
import numpy as np
import glob
def FindLatestModel(CheckPointPath):
    FileList = glob.glob(CheckPointPath + '*.ckpt.index') # * means all if need specific format then *.csv
    LatestFile = max(FileList, key=os.path.getctime)
    # Strip everything else except needed information
    LatestFile = LatestFile.replace(CheckPointPath, '')
    LatestFile = LatestFile.replace('.ckpt.index', '')
    return LatestFile

def SetupAll(args):
    # Setup DirNames
    DirNamesPath = f"{args.data_list}/{args.Dataset}_dirnames.txt"
    TrainPath = f"{args.data_list}/{args.Dataset}_train.txt"
    ValPath = f"{args.data_list}/{args.Dataset}_val.txt"
    TestPath = f"{args.data_list}/{args.Dataset}_test.txt"
    DirNames, TrainNames, ValNames, TestNames=\
              ReadDirNames(DirNamesPath, TrainPath, ValPath, TestPath)


    # Setup Neural Net Params
    # List of all OptimizerParams: depends on Optimizer
    # For ADAM Optimizer: [LearningRate, Beta1, Beta2, Epsilion]
    UseDefaultFlag = 0 # Set to 0 to use your own params, do not change default parameters
    if UseDefaultFlag:
        # Default Parameters
        OptimizerParams = [1e-3, 0.9, 0.999, 1e-8]
    else:
        # Custom Parameters
        OptimizerParams = [args.LR, 0.9, 0.999, 1e-8]   
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 1000 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    if args.Dataset == 'FT3D':
        OriginalImageSize = np.array([540, 960, 3])
        PatchSize = np.array([480, 640, 3])
        NumOut = 2
    elif args.Dataset == 'FC2':
        OriginalImageSize = np.array([384, 512, 3])
        PatchSize = np.array([352, 480, 3])
        NumOut = 2
    elif args.Dataset == 'MSCOCO':
        OriginalImageSize = np.array([504,640,3])
        PatchSize = np.array([352, 480, 3])
        NumOut = 3
    NumTrainSamples = len(TrainNames)
    NumValSamples = len(ValNames)
    NumTestSamples = len(TestNames)
    Lambda = np.array([1.0, 1.0]) # Loss, Reg

    # Pack everything into args
    args.TrainNames = TrainNames
    args.ValNames = ValNames
    args.TestNames = TestNames
    args.OptimizerParams = OptimizerParams
    args.SaveCheckPoint = SaveCheckPoint
    args.PatchSize = PatchSize
    args.NumTrainSamples = NumTrainSamples
    args.NumValSamples = NumValSamples
    args.NumTestSamples = NumTestSamples
    args.NumTestRunsPerEpoch = NumTestRunsPerEpoch
    args.OriginalImageSize = OriginalImageSize
    args.Lambda = Lambda
    args.NumOut = NumOut
    args.LogsPath = "/experiments/logs//"
    args.CheckPointPath = "/experiments/models//"
    args.LossFuncName = "MultiscaleSL1-1"
    
    if(not (os.path.isdir(args.CheckPointPath))):
       os.makedirs(args.CheckPointPath)
    if(not (os.path.isdir(args.LogsPath))):
       os.makedirs(args.LogsPath)
    
    if args.LoadCheckPoint==1:
        args.LatestFile = FindLatestModel(args.CheckPointPath)
    else:
        args.LatestFile = None
        
    return args


def ReadDirNames(DirNamesPath, TrainPath, ValPath, TestPath):
    """
    Inputs: 
    Path is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read DirNames and LabelNames files
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    
    # Read Train, Val and Test Idxs
    TrainIdxs = open(TrainPath, 'r')
    TrainIdxs = TrainIdxs.read()
    TrainIdxs = TrainIdxs.split()
    TrainIdxs = [int(val) for val in TrainIdxs]
    TrainNames = [DirNames[i] for i in TrainIdxs]
    # TrainLabels = [LabelNames[i] for i in TrainIdxs]

    ValIdxs = open(ValPath, 'r')
    ValIdxs = ValIdxs.read()
    ValIdxs = ValIdxs.split()
    ValIdxs = [int(val) for val in ValIdxs]
    ValNames = [DirNames[i] for i in ValIdxs]
    # ValLabels = [LabelNames[i] for i in ValIdxs]

    TestIdxs = open(TestPath, 'r')
    TestIdxs = TestIdxs.read()
    TestIdxs = TestIdxs.split()
    TestIdxs = [int(val) for val in TestIdxs]
    TestNames = [DirNames[i] for i in TestIdxs]
    # TestLabels = [LabelNames[i] for i in TestIdxs]

    return DirNames, TrainNames, ValNames, TestNames
