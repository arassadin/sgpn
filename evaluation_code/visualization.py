"""
    Simple Usage example (with 3 images)
"""
import os
import math
import numpy as np
import argparse
import random
import pickle
import torch
import scipy.misc as misc
import csv
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from shutil import copyfile

sys.path.append('.')

from torch.autograd import Variable


ID_COLOR = {
 0 : [0,   0,   0  ],
 1 : [174, 199, 232],
 2 : [152, 223, 138],
 3 : [31,  119, 180],
 4 : [255, 187, 120],
 5 : [188, 189, 34 ],
 6 : [140, 86,  75 ],
 7 : [255, 152, 150],
 8 : [214, 39,  40 ],
 9 : [197, 176, 213],
 10: [148, 103, 189],
 11: [196, 156, 148],
 12: [23,  190, 207],
 13: [178, 76,  76 ],
 14: [247, 182, 210],
 15: [66,  188, 102],
 16: [219, 219, 141],
 17: [140, 57,  197],
 18: [202, 185, 52 ],
 19: [51,  176, 203],
 20: [200, 54,  131],
 21: [92,  193, 61 ],
 22: [78,  71,  183],
 23: [172, 114, 82 ],
 24: [255, 127, 14 ],
 25: [91,  163, 138],
 26: [153, 98,  156],
 27: [140, 153, 101],
 28: [158, 218, 229],
 29: [100, 125, 154],
 30: [178, 127, 135],
 31: [120, 185, 128],
 32: [146, 111, 194],
 33: [44,  160, 44 ],
 34: [112, 128, 144],
 35: [96,  207, 209],
 36: [227, 119, 194],
 37: [213, 92,  176],
 38: [94,  106, 211],
 39: [82,  84,  163],
 40: [100, 85,  144]}

ID_LABEL = {
        0 : "empty",          
        1 : "cabinet",         
        2 : "bed",             
        3 : "chair",           
        4 : "sofa",            
        5 : "table",           
        6 : "door",            
        7 : "window",          
        8 : "bookshelf",       
        9 : "counter",            
        10: "desk",         
        11: "curtain",         
        12: "refrigerator",         
        13: "shower_curtain",          
        14: "toilet",      
        15: "sink",           
        16: "bathtub",     
        17: "otherfurniture",          
        18: "ignore",            
        }


'''
ID_LABEL = {
        0 : "ignore",          
        1 : "cabinet",         
        2 : "bed",             
        3 : "chair",           
        4 : "sofa",            
        5 : "table",           
        6 : "door",            
        7 : "window",          
        8 : "bookshelf",       
        9 : "counter",         
        10: "desk",            
        11: "shelves",         
        12: "curtain",         
        13: "dresser",         
        14: "mirror",          
        15: "television",      
        16: "towel",           
        17: "night_stand",     
        18: "toilet",          
        19: "sink",            
        20: "lamp",            
        21: "bathtub",         
        22: "bag",             
        23: "otherstructure",  
        24: "otherfurniture",  
        25: "otherprop"
        }
'''


def label2rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in range(41):
        mask = label_img == l
        label_img_rgb[mask] = ID_COLOR[l]

    return label_img_rgb.astype(np.uint8)



def make_box_mesh(box_min, box_max): 
    vertices = [
        np.array([box_max[0], box_max[1], box_max[2]]),
        np.array([box_min[0], box_max[1], box_max[2]]),
        np.array([box_min[0], box_min[1], box_max[2]]),
        np.array([box_max[0], box_min[1], box_max[2]]),
        np.array([box_max[0], box_max[1], box_min[2]]),
        np.array([box_min[0], box_max[1], box_min[2]]),
        np.array([box_min[0], box_min[1], box_min[2]]),
        np.array([box_max[0], box_min[1], box_min[2]])
    ]
    indices = [
        np.array([1, 2, 3], dtype=np.uint32),
        np.array([1, 3, 0], dtype=np.uint32),
        np.array([0, 3, 7], dtype=np.uint32),
        np.array([0, 7, 4], dtype=np.uint32),
        np.array([3, 2, 6], dtype=np.uint32),
        np.array([3, 6, 7], dtype=np.uint32),
        np.array([1, 6, 2], dtype=np.uint32),
        np.array([1, 5, 6], dtype=np.uint32),
        np.array([0, 5, 1], dtype=np.uint32),
        np.array([0, 4, 5], dtype=np.uint32),
        np.array([6, 5, 4], dtype=np.uint32),
        np.array([6, 4, 7], dtype=np.uint32)
    ]
    return vertices, indices


def save_mesh(verts, indices, output_file, color=None):
    with open(output_file, 'w') as f:
        for v in verts:
	        f.write('v %f %f %f %f %f %f %f\n' % (v[0], v[1], v[2], color[0], color[1], color[2], 0.5))
        f.write('g foo\n')
        for ind in indices:
            f.write('f %d %d %d\n' % (ind[0] + 1, ind[1] + 1, ind[2] + 1))
        f.write('g\n')


def get_bbox_verts(bbox_min, bbox_max):
    verts = [
        np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

        np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
    ]
    return verts

def get_bbox_edges(bbox_min, bbox_max):
    box_verts = get_bbox_verts(bbox_min, bbox_max)
    edges = [
        (box_verts[0], box_verts[1]),
        (box_verts[1], box_verts[2]),
        (box_verts[2], box_verts[3]),
        (box_verts[3], box_verts[0]),

        (box_verts[4], box_verts[5]),
        (box_verts[5], box_verts[6]),
        (box_verts[6], box_verts[7]),
        (box_verts[7], box_verts[4]),

        (box_verts[0], box_verts[4]),
        (box_verts[1], box_verts[5]),
        (box_verts[2], box_verts[6]),
        (box_verts[3], box_verts[7])
    ]
    return edges

def rotation(axis, angle):
    rot = np.eye(4)
    c = np.cos(-angle)
    s = np.sin(-angle)
    t = 1.0 - c
    axis /= compute_length_vec3(axis)
    x = axis[0]
    y = axis[1]
    z = axis[2]
    rot[0,0] = 1 + t*(x*x-1)
    rot[0,1] = z*s+t*x*y
    rot[0,2] = -y*s+t*x*z
    rot[1,0] = -z*s+t*x*y
    rot[1,1] = 1+t*(y*y-1)
    rot[1,2] = x*s+t*y*z
    rot[2,0] = y*s+t*x*z
    rot[2,1] = -x*s+t*y*z
    rot[2,2] = 1+t*(z*z-1)
    return rot

def compute_length_vec3(vec3):
    return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])

def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
    verts = []
    indices = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)
    for i in range(stacks+1):
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
            verts.append(pos)
    for i in range(stacks):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]
            if (math.fabs(dotx) != 1.0):
                axis = np.array([1,0,0]) - dotx * va
            else:
                axis = np.array([0,1,0]) - va[1] * va
            axis /= compute_length_vec3(axis)
        transform = rotation(axis, -angle)
    transform[:3,3] += p0
    verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
    verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
        
    return verts, indices

def create_bbox_mesh(bbox, output_file, radius=0.2, offset=np.array([0, 0, 0]), color=None):
    bbox_min = np.array([bbox[0], bbox[1], bbox[2]])
    bbox_max = np.array([bbox[3], bbox[4], bbox[5]])
    edges = get_bbox_edges(bbox_min, bbox_max)
    verts = []
    indices = []
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
    save_mesh(verts, indices, output_file, ID_COLOR[int(color)])


def load_mapping(label_file):
    mapping = dict()
    csvfile = open(label_file) 
    spamreader = csv.DictReader(csvfile, delimiter=',')
    for row in spamreader:
        mapping[int(row['nyu40id'])] = int(row['mappedIdConsecutive'])
    csvfile.close()
    mapping[0] = 0
    return mapping

def mapping_to(image, mapping):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j] = mapping[image[i,j]]
    return image

def save_color_mesh(verts, indices, output_file, color=None):
    with open(output_file, 'w') as f:
        for v, c in zip(verts, color):
	        f.write('v %f %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2], 0.5))
        f.write('g foo\n')
        for ind in indices:
            f.write('f %d %d %d\n' % (ind[0] + 1, ind[1] + 1, ind[2] + 1))
        f.write('g\n')


def create_mask_mesh(binary_grid, output_file, offset=np.array([0, 0, 0]), scale=np.array([1, 1, 1]), color=None):
    verts = []
    indices = []
    colors = []
    for z in range(binary_grid.shape[2]):
        for y in range(binary_grid.shape[1]):
            for x in range(binary_grid.shape[0]):
                if binary_grid[x, y, z] > 0:
                    box_min = (np.array([x, y, z]) + 0.05)*scale + offset
                    box_max = (np.array([x, y, z]) + 0.95)*scale + offset
                    box_verts, box_ind = make_box_mesh(box_min, box_max)
                    cur_num_verts = len(verts)
                    box_ind = [x + cur_num_verts for x in box_ind]
                    color_mesh = [ID_COLOR[int(binary_grid[x,y,z])] for i in range(8)] if color==None  \
                            else [ID_COLOR[int(color)] for i in range(8)]
                    verts.extend(box_verts)
                    indices.extend(box_ind)
                    colors.extend(color_mesh)
    save_color_mesh(verts, indices, output_file, color=colors)


def visualize(output_dir, data=None, bbox=None, mask=None, color=None):
    os.makedirs(output_dir, exist_ok=True)
    if data is not None:
        create_mask_mesh(data, os.path.join(output_dir, 'data.obj'))
    if bbox is not None:
        for idx, box in enumerate(bbox):
            try:
                create_bbox_mesh(box, os.path.join(output_dir, 'bbox_' + str(idx) + '.obj'), color=box[6])
            except:
                create_bbox_mesh(box, os.path.join(output_dir, 'bbox_' + str(idx) + '.obj'), color=1)
    if mask is not None:
        for idx, mask_item in enumerate(mask):
            try:
                create_mask_mesh(mask_item, os.path.join(output_dir, 'mask_' + str(idx) + '.obj'), offset=bbox[idx][:3], color=bbox[idx][6])
            except:
                create_mask_mesh(mask_item, os.path.join(output_dir, 'mask_' + str(idx) + '.obj'), offset=bbox[idx][:3], color=0)
    if color is not None:
        create_mask_mesh(color, os.path.join(output_dir, 'color.obj'))
