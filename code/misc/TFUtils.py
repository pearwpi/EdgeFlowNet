import os, pdb
import sys
import time
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Don't generate pyc codes
sys.dont_write_bytecode = True

def FindNumParams(PrintFlag=None):
    if(PrintFlag is not None):
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

def FindNumFlops(sess, PrintFlag=None):
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
    if(PrintFlag is not None):
        print('Number of Flops in this model are %d' % flops.total_float_ops)
    return flops.total_float_ops

def SetGPU(GPUNum=-1):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUNum)

def CalculateModelSize(PrintFlag=None):
    var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
                 for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)]
    # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    # print(var_sizes)       
    # input('q')
    SizeMB = sum(var_sizes) / (1024. ** 2)
    if(PrintFlag is not None):
        print('Expected Model Size is %f' % SizeMB)
    return SizeMB
            
def Rename(CheckPointPath, ReplaceSource=None, ReplaceDestination=None, AddPrefix=None, AddSuffix=None):
    # Help!
    # https://github.com/tensorflow/models/issues/974
    # https://stackoverflow.com/questions/37086268/rename-variable-scope-of-saved-model-in-tensorflow
    # https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
    # Rename to correct paths in checkpoint file
    # CheckPointPath points to folder with all the model files and checkpoint file
    if(not os.path.isdir(CheckPointPath)):
        print('CheckPointsPath should be a directory!')
        os._exit(0)
        
    CheckPoint = tf.train.get_checkpoint_state(CheckPointPath)
    with tf.Session() as sess:
        for VarName, _ in tf.contrib.framework.list_variables(CheckPointPath):
            # Load the variable
            Var = tf.contrib.framework.load_variable(CheckPointPath, VarName)

            NewName = VarName
            if(ReplaceSource is not None):
                NewName = NewName.replace(ReplaceSource, ReplaceDestination)
            if(AddPrefix is not None):
                NewName = AddPrefix + NewName
            if(AddSuffix is not None):
                NewName = NewName + AddSuffix

            print('Renaming %s to %s.' % (VarName, NewName))
            # Rename the variable
            Var = tf.Variable(Var, name=NewName)
            
        # Save the variables
        Saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        Saver.save(sess, CheckPoint.model_checkpoint_path)

def PrintVars(CheckPointPath):
    # Help!
    # https://github.com/tensorflow/models/issues/974
    # https://stackoverflow.com/questions/37086268/rename-variable-scope-of-saved-model-in-tensorflow
    # https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
    # Rename to correct paths in checkpoint file
    # CheckPointPath points to folder with all the model files and checkpoint file
    if(not os.path.isdir(CheckPointPath)):
        print('CheckPointsPath should be a directory!')
        os._exit(0)
        
    CheckPoint = tf.train.get_checkpoint_state(CheckPointPath)
    with tf.Session() as sess:
        for VarName, _ in tf.contrib.framework.list_variables(CheckPointPath):
            print('%s' % VarName)


def freeze_graph(model_dir, output_node_names):
    # Taken from: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 

    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def
