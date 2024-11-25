#!/usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps

# Don't generate pyc codes
sys.dont_write_bytecode = True

def Count(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        self.CurrBlock += 1
        return func(self, *args, **kwargs)
    return wrapped

def CountAndScope(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        with tf.compat.v1.variable_scope(func.__name__ + str(self.CurrBlock) + str(self.Suffix)):
            self.CurrBlock += 1
            return func(self, *args, **kwargs)
    return wrapped

def Scope(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        with tf.compat.v1.variable_scope(func.__name__):
            return func(self, *args, **kwargs)
    return wrapped
    
# def CountAndScope(func):
#     @wraps(func)
#     def wrapped(self, *args, **kwargs):
#         with tf.variable_scope(func.__name__ + str(self.CurrBlock)):
#             self.CurrBlock += 1
#             return func(self, *args, **kwargs)
#     return wrapped

# def Scope(func):
#     @wraps(func)
#     def wrapped(self, *args, **kwargs):
#         with tf.variable_scope(func.__name__):
#             return func(self, *args, **kwargs)
#     return wrapped


# def CountAndScope(func=None, Suffix=None):
#     @wraps(func)
#     def wrapped(self, *args, **kwargs):
#         if(Suffix is not None):
#             Name = func.__name__ + str(self.CurrBlock) + Suffix
#         else:
#             Name = func.__name__ + str(self.CurrBlock)
#         with tf.variable_scope(Name):
#             self.CurrBlock += 1
#             return func(self, *args, **kwargs)
#     return wrapped

# def Scope(func=None, Suffix=None):
#     @wraps(func)
#     def wrapped(self, *args, **kwargs):
#         if(Suffix is not None):
#             Name = func.__name__ + Suffix
#         else:
#             Name = func.__name__ 
#         with tf.variable_scope(Name):
#             return func(self, *args, **kwargs)
#     return wrapped
