#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math
import cv2

def normalize(qty, max_val, min_val):
    return ((qty - min_val) / (max_val - min_val))*255.0

def denormalize(qty, max_val, min_val):
    return (qty/255.0)*(max_val - min_val) + min_val

def shift_and_scale_angle(qty):
    """
    shift by np.pi and scale from radians to degrees 
    """
    return (qty + np.pi)*180.0/np.pi

def deshift_and_descale_angle(qty):
    """
    descale from degrees to radians and deshift by np.pi
    """
    return (qty*np.pi)/180.0 - np.pi

def flow_to_polar_np(flow_xy, flow_r_max = 50.0, flow_r_min = -50.0, flow_t_max = 360.0, flow_t_min = 0.0):
    """
    takes cartesian and gives out polar
    flow_xy - H x W x 2
    flow_r_max, flow_r_min, flow_t_max, flow_t_min - normalization values
    flow_rt - H x W x 2 which is normalized

    """
    mag = np.sqrt(flow_xy[:,:,0]**2 + flow_xy[:,:,1]**2)
    ang = np.arctan2(flow_xy[:,:,1], flow_xy[:,:,0])
    ang = shift_and_scale_angle(ang)
    mag_norm = normalize(mag, flow_r_max, flow_r_min)
    ang_norm = normalize(ang, flow_t_max, flow_t_min)
    flow_rt = np.stack([mag_norm, ang_norm], axis = 2)
    return flow_rt

def flow_from_polar_np(flow_pt, flow_r_max = 50.0, flow_r_min = -50.0, flow_t_max = 360.0, flow_t_min = 0.0):
    """
    takes polar predictions and gives out cartesian flow
    flow_polar - H x W x 2
    flow_r_max, flow_r_min, flow_t_max, flow_t_min - normalization values
    flow_xy - H x W x 2 which is denormalized
    """
    mag = flow_pt[:, :, 0]
    ang = flow_pt[:, :, 1]
    mag_denorm = denormalize(mag, flow_r_max, flow_r_min)
    ang_denorm = denormalize(ang, flow_t_max, flow_t_min)
    ang_denorm = deshift_and_descale_angle(ang_denorm)
    flow_x = np.cos(ang_denorm)*mag_denorm
    flow_y = np.sin(ang_denorm)*mag_denorm
    flow_xy = np.stack([flow_x, flow_y], axis = 2)
    return flow_xy

def flow_to_polar_tf(flow_xy, flow_r_max = 50.0, flow_r_min = -50.0, flow_t_max = 360.0, flow_t_min = 0.0):
    """
    takes cartesian and gives out polar
    flow_xy - B x H x W x 2
    flow_r_max, flow_r_min, flow_t_max, flow_t_min - normalization values
    flow_rt - B x H x W x 2 which is normalized
    """
    mag = tf.compat.v1.math.sqrt(flow_xy[:,:,:,0]**2 + flow_xy[:,:,:,1]**2)
    ang = tf.compat.v1.math.atan2(flow_xy[:,:,:,1], flow_xy[:,:,:,0])
    ang = shift_and_scale_angle(ang)
    mag_norm = normalize(mag, flow_r_max, flow_r_min)
    ang_norm = normalize(ang, flow_t_max, flow_t_min)
    flow_rt = tf.compat.v1.stack([mag_norm, ang_norm], axis = 3)
    return flow_rt


def flow_from_polar_tf(flow_pt, flow_r_max = 50.0, flow_r_min = -50.0, flow_t_max = 360.0, flow_t_min = 0.0):
    """
    takes polar predictions and gives out cartesian flow
    flow_polar - B x H x W x 2
    flow_r_max, flow_r_min, flow_t_max, flow_t_min - normalization values
    flow_xy - B x H x W x 2 which is denormalized
    """
    mag = flow_pt[:, :, :, 0]
    ang = flow_pt[:, :, :, 1]
    mag_denorm = denormalize(mag, flow_r_max, flow_r_min)
    ang_denorm = denormalize(ang, flow_t_max, flow_t_min)
    ang_denorm = deshift_and_descale_angle(ang_denorm)
    flow_x = tf.compat.v1.math.cos(ang_denorm)*mag_denorm
    flow_y = tf.compat.v1.math.sin(ang_denorm)*mag_denorm
    flow_xy = tf.compat.v1.stack([flow_x, flow_y], axis = 3)
    return flow_xy


def tests():
    test_angles = np.array([np.pi/2,-np.pi/2, 3*np.pi/4, -3*np.pi/4])
    scaled_and_shifted = shift_and_scale_angle(test_angles)
    descaled_and_deshifted = deshift_and_descale_angle(scaled_and_shifted)
    normalized_angles = normalize(scaled_and_shifted, 360.0, 0.0)
    denormalized_angles = denormalize(normalized_angles, 360.0, 0.0)
    assert(np.all(descaled_and_deshifted == test_angles))
    assert(np.all(denormalized_angles == scaled_and_shifted))
    print(f"test_angles:{test_angles}")
    print(f"scaled_and_shifted:{scaled_and_shifted}")
    print(f"descaled_and_deshifted:{descaled_and_deshifted}")
    print(f"normalized_angles:{normalized_angles}")
    print(f"denormalized_angles:{denormalized_angles}")