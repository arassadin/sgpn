import os
import numpy as np
import csv
import json
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
from timer import timer


filename = 'scannet_all.txt'
scan_path = '/mnt/braxis_scannet/public/v2/scans/'


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

def point_label_to_obj(points, out_filename, show_color='data'):
    """ For visualization of a room from data_label file,
    input_filename: each line is X Y Z R G B L I
    out_filename: OBJ filename,
            visualize input file by coloring point with label color
        easy_view: only visualize furnitures and floor
    """
    data = points[:, 0:6]
    label = points[:, -2].astype(int)
    instance = points[:, -1].astype(int)
    fout = open(out_filename, 'w')
    for i in range(data.shape[0]):
        if show_color == 'label':
            color = ID_COLOR[label[i]]
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], color[0], color[1], color[2]))
        elif show_color == 'data':
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], data[i, 3], data[i, 4], data[i, 5]))
        elif show_color == 'instance':
            color = ID_COLOR[instance[i]%41]
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], color[0], color[1], color[2]))
    fout.close()


def get_label2id(filename):
    mapping = dict()
    csvfile = open(filename) 
    spamreader = csv.DictReader(csvfile, delimiter='\t')
    for row in spamreader:
        #[('nyu40id', '1'), ('nyu40class', 'wall'),]
        mapping[row['raw_category']] = int(row['nyu40id'])
    csvfile.close()

    return mapping

scenes = open(filename).readlines()
label2id = get_label2id('scannetv2-labels.combined.tsv')

timer.tic()
for scene in scenes:
    print(scene.strip())
    if os.path.exists('scannet_data/annotation/{}.npy'.format(scene.strip())):
        continue
    scene_path = os.path.join(scan_path, scene.strip())

    ply_file = os.path.join(scene_path, scene.strip() + '_vh_clean_2.ply')
    with open(ply_file, 'rb') as read_file:
        ply_data = PlyData.read(read_file)

    #dict_keys(['sceneId', 'appId', 'segGroups', 'segmentsFile'])
    instance_file = os.path.join(scene_path, scene.strip() + '.aggregation.json')
    with open(instance_file, 'r') as read_file:
        instance_data = json.load(read_file)

    #dict_keys(['params', 'sceneId', 'segIndices'])
    seg_file = os.path.join(scene_path, instance_data['segmentsFile'][8:] if 'scannet' in instance_data['segmentsFile'] else instance_data['segmentsFile'])
    with open(seg_file, 'r') as read_file:
        seg_data = json.load(read_file)

    points = []
    for x,y,z,r,g,b,_ in ply_data['vertex']:
        points.append([x,y,z,r,g,b, 0, -1])
    points = np.array(points)

    #dict_keys(['id', 'objectId', 'segments', 'label'])
    for instance in instance_data['segGroups']:
        instance_id = instance['id']
        label_id = label2id[instance['label']]
        for point_index, seg_label in enumerate(seg_data['segIndices']):
            if seg_label in instance['segments']:
                points[point_index][6] = label_id
                points[point_index][7] = instance_id
    #point_label_to_obj(points, 'data.obj', show_color='data')
    #point_label_to_obj(points, 'instance.obj', show_color='instance')
    #point_label_to_obj(points, 'label.obj', show_color='label')
    np.save('scannet_data/annotation/{}'.format(scene.strip()), points)
timer.toc()
print(timer.total_time())

