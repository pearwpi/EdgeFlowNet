#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math
import cv2

# Taken from: https://github.com/daniilidis-group/EV-FlowNet
"""
Generates an RGB image where each point corresponds to flow in that direction from the center,
as visualized by flow_viz_tf.
Output: color_wheel_rgb: [1, width, height, 3]
"""
def draw_color_wheel_np(width, height):
    color_wheel_x = np.linspace(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = np.linspace(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = np.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_rgb = flow_viz_np(color_wheel_X, color_wheel_Y)
    return color_wheel_rgb


"""
Visualizes optical flow in HSV space using TensorFlow, with orientation as H, magnitude as V.
Returned as RGB.
Input: flow: [batch_size, width, height, 2]
Output: flow_rgb: [batch_size, width, height, 3]
"""



def flow_viz_np(flow_x, flow_y):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

def flow_viz_np2(flow_x, flow_y):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_y.shape[1], 3], dtype=np.uint8) # only flow_y shape is different from above
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb


def flow_viz_np3(flows):
    flow_x = flows[:, :, 0]
    flow_y = flows[:, :, 1]
    return flow_viz_np(flow_x, flow_y)


def flow_mag(flow_x, flow_y):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)
    return mag

def flow_polar(flow_x, flow_y):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_y.shape[1], 2], dtype=np.uint8) # only flow_y shape is different from above
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return hsv

def flow_viz_polar_tf(flow_polar):
    ones = tf.ones([flow_polar.shape[0],flow_polar.shape[1],flow_polar.shape[2]])
    hsv = tf.stack([flow_polar[:,:,:,0], ones, flow_polar[:,:,:,1]], axis=3)
    flow_rgb = tf.image.hsv_to_rgb(hsv)
    return flow_rgb

def flow_viz_polar_np(flow_polar_ang, flow_polar_mag):
    ones = np.ones([flow_polar_ang.shape[0],flow_polar_ang.shape[1]], dtype=np.uint8)
    hsv = np.stack([flow_polar_ang, ones, flow_polar_mag], axis=2)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

def flow_unviz_np(flow_rgb):
    flow_hsv = cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2HSV).astype(np.float64)
    ang = flow_hsv[:, :, 0]
    mag = flow_hsv[:, :, 2]
    ang *= np.pi*2/180
    ang -= np.pi
    flow_x = np.cos(ang)*mag
    flow_y = np.sin(ang)*mag
    flow = np.stack([flow_x, flow_y], axis=2)
    return flow

def flow_unviz_polar_np(flow_hsv):
    ang = flow_hsv[:, :, 0]
    mag = flow_hsv[:, :, 1]
    ang *= np.pi*2/180
    ang -= np.pi
    flow_x = np.cos(ang)*mag
    flow_y = np.sin(ang)*mag
    flow = np.stack([flow_x, flow_y], axis=2)
    return flow

