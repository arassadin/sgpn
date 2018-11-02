# Example of the output format for evaluation for 3d semantic label and instance prediction.
# Exports a train scan in the evaluation format using:
#   - the *_vh_clean_2.ply mesh
#   - the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files
#
# example usage: export_train_mesh_for_evaluation.py --scan_path [path to scan data] --output_file [output file] --type label
# Note: technically does not need to load in the ply file, since the ScanNet annotations are defined against the mesh vertices, but we load it in here as an example.

# python imports
import math
import os, sys, argparse
import inspect
import json
import csv
import pickle
from BinaryReader import gt_reader

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import util
import util_3d

padding = [10, 16, 10, 0]

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

def save_merged(merged_rois, merged_masks, merged_class, merged_scores, save_path):
    f = open(save_path, 'wb')
    save_dict = {
        'rois': merged_rois,
        'masks': merged_masks,
        'class': merged_class,
        'scores': merged_scores
    }
    pickle.dump(save_dict, f)
    print("{} done".format(save_path))

def save_3d_points(verts, output_file, instance=None):
    with open(output_file, 'w') as f:
        f.write("ply\n");
        f.write("format ascii 1.0\n");
        f.write("element vertex {:d}\n".format(len(verts)));
        f.write("property float x\n");
        f.write("property float y\n");
        f.write("property float z\n");
        f.write("property uchar red\n");
        f.write("property uchar green\n");
        f.write("property uchar blue\n");
        f.write("end_header\n");
        if instance == None:
            for v in verts:
                f.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(v[0], v[1], v[2], 0, 0, 0))
        else:
            for v, i in zip(verts, instance):
                f.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(v[0], v[1], v[2], ID_COLOR[i%41][0], ID_COLOR[i%41][1], ID_COLOR[i%41][2]))

def load_pred(pred_file):
    instances = []
    with open(pred_file) as f:
        lines = f.readlines()
        for line in lines:
            info = line.split()

            filename = info[0][7:]

            cls = int(info[1])
            conf = float(info[2])
            vertics = []

            with open(os.path.join(os.path.dirname(pred_file), filename)) as f1:
                vertices = f1.readlines()

            instances.append({
                'class': cls,
                'scores': conf,
                'vertices': vertices
                })
    return instances

def load_gt(gt_file='/mnt/raid/ji/mask_rcnn_2dto3d/samples/scannet/gt/scene0644_00.txt'):
    instance_label = []
    with open(gt_file) as f:
        lines = f.readlines()
        for line in lines:
            # %: instance, /: label
            instance_label.append(int(int(line.strip()) / 1000))
    return instance_label

def export(mesh_vertices, world2grid, instances, blob, output_file):
    ignore_class = [0, 1, 20]
    class_mapping = {
            2:1,
            3:2,
            4:3,
            5:4,
            6:5,
            7:6,
            8:7,
            9:8,
            10:9,
            11:10, 
            12:11,
            13:12,
            14:13,
            15:14,
            16:15,
            17:16,
            18:17,
            19:18
            }
    rois = {}
    scores = {}
    masks = {}
    classes = {}

    
    instance_id = 1
    for instance in instances:
        if instance['class'] in ignore_class:
            continue
        grid_coords = []
        minx = 9999
        miny = 9999
        minz = 9999
        maxx = 0
        maxy = 0
        maxz = 0
        counter = 0

        for ind, vertex in enumerate(mesh_vertices):
            if instance['vertices'][ind] == '0\n':
                continue
            else:
                grid_coord = np.round(np.matmul(world2grid, np.append(vertex, 1)))
                if grid_coord[1] < 48:
                    counter += 1
                    grid_coords.append(grid_coord)
                    if grid_coord[0] < minx:
                        minx = grid_coord[0]
                    if grid_coord[1] < miny:
                        miny = grid_coord[1]
                    if grid_coord[2] < minz:
                        minz = grid_coord[2]
                    if grid_coord[0] > maxx:
                        maxx = grid_coord[0]
                    if grid_coord[1] > maxy:
                        maxy = grid_coord[1]
                    if grid_coord[2] > maxz:
                        maxz = grid_coord[2]

        if counter == 0:
            print('nonvalid')
            continue

        mask = np.zeros((int(maxx-minx+1), int(maxy-miny+1), int(maxz-minz+1)))
        min_dist = 2
        for mask_i in range(mask.shape[0]):
            for mask_j in range(mask.shape[1]):
                for mask_k in range(mask.shape[2]):
                    if blob['data'][0][int(mask_i+minx), int(mask_j+miny), int(mask_k+minz)] <= 1.0:
                        for grid_coord in grid_coords:
                            dist = (grid_coord[0]-minx-mask_i)**2 + (grid_coord[1]-miny-mask_j)**2 + (grid_coord[2]-minz-mask_k) ** 2
                            if dist < min_dist:
                                mask[mask_i, mask_j, mask_k] = 1
                                break

        rois[instance_id] = [minx, miny, minz, maxx+1, maxy+1, maxz+1]
        scores[instance_id] = instance['scores']
        masks[instance_id] = mask
        classes[instance_id] = class_mapping[instance['class']]
        instance_id += 1

    save_merged(rois, masks, classes, scores, output_file)
            

def load_matrix(filename):
    matrix = np.zeros((4, 4))
    with open(filename) as f:
        lines = f.readlines()
    for ind, line in enumerate(lines):
        matrix[ind][0] = float(line.split()[0])
        matrix[ind][1] = float(line.split()[1])
        matrix[ind][2] = float(line.split()[2])
        matrix[ind][3] = float(line.split()[3]) - padding[ind]
    return  matrix

def save_color_mesh(verts, indices, output_file, color=None):
    with open(output_file, 'w') as f:
        for v, c in zip(verts, color):
	        f.write('v %f %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2], 0.5))
        f.write('g foo\n')
        for ind in indices:
            f.write('f %d %d %d\n' % (ind[0] + 1, ind[1] + 1, ind[2] + 1))
        f.write('g\n')

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

def create_mask_mesh(grid, output_file, offset=np.array([0, 0, 0]), scale=np.array([1, 1, 1])):
    verts = []
    indices = []
    colors = []
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                if grid[x, y, z] > 0:
                    box_min = (np.array([x, y, z]) + 0.05)*scale + offset
                    box_max = (np.array([x, y, z]) + 0.95)*scale + offset
                    box_verts, box_ind = make_box_mesh(box_min, box_max)
                    cur_num_verts = len(verts)
                    box_ind = [x + cur_num_verts for x in box_ind]
                    color_mesh = [ID_COLOR[int(grid[x,y,z])] for i in range(8)]
                    verts.extend(box_verts)
                    indices.extend(box_ind)
                    colors.extend(color_mesh)
    save_color_mesh(verts, indices, output_file, color=colors)

def visualize_point(mesh_vertices, world2grid, instances):
    ignore_class = [0,1,20]
    grid_coords = []
    pred_instance = []
    for ind, vertex in enumerate(mesh_vertices):
        grid_coord = np.floor(np.matmul(world2grid, np.append(vertex, 1)))
        grid_coords.append(grid_coord)
        instance_id = 0
        for instance_ind, instance in enumerate(instances):
            if instance['class']  in ignore_class:
                continue
            if instance['vertices'][ind] == '1\n':
                instance_id = instance_ind
        pred_instance.append(instance_id)
    save_3d_points(grid_coords, 'pred_point.ply', pred_instance)

def visualize_voxel(mesh_vertices, world2grid, instances, blob):
    ignore_class = [0,1,20]
    gt_grid = np.zeros(blob['dim'])
    for ind, box in enumerate(blob['gt_box']):
        gt_grid[box[0]:box[3], box[1]:box[4], box[2]:box[5]] = blob['gt_mask'][ind] * box[6]
    create_mask_mesh(gt_grid, 'gt_voxel.obj')
    
    pred_grid = np.zeros(blob['dim'])
    for ind, vertex in enumerate(mesh_vertices):
        grid_coord = np.ceil(np.matmul(world2grid, np.append(vertex, 1)))
        for instance_ind, instance in enumerate(instances):
            if instance['class']  in ignore_class:
                continue
            if instance['vertices'][ind] == '1\n':
                instance_id = instance['class']
                if grid_coord[1] < 48 and blob['data'][0][int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2])] <= 1:
                    pred_grid[int(grid_coord[0]), int(grid_coord[1]), int(grid_coord[2])] = instance_id
    create_mask_mesh(pred_grid, 'pred_voxel.obj')
    create_mask_mesh(np.where(blob['data'][0]<=1, 1, 0), 'data.obj')

def visualize(mesh_vertices, world2grid, instance, blob):
    # visualize point
    visualize_point(mesh_vertices, world2grid, instance)
    # visualize voxels
    visualize_voxel(mesh_vertices, world2grid, instance, blob)

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', default='/mnt/raid/ji/SGPN/output')
parser.add_argument('--output_dir', default="/mnt/raid/ji/SGPN/SGPN_merged", help='output file')
parser.add_argument('--scan_path', default='/mnt/raid/ji/SGPN/mesh')
parser.add_argument('--frames', default='/mnt/braxis_datasets/ScanNet/frames_square')
parser.add_argument('--groundtruth_path', default="/mnt/local_datasets/Voxelization/ji/ScanNet/vox-5cm_sdf/test/scene")

opt = parser.parse_args()


def main():
    for ind, pred_file in enumerate(os.listdir(opt.pred_dir)):
        if not os.path.isfile(os.path.join(opt.pred_dir, pred_file)):
            continue
        scan_name = os.path.splitext(pred_file)[0]
        mesh_file = os.path.join(opt.scan_path, scan_name + '.ply')
        if not os.path.exists(mesh_file):
            continue
        print(pred_file)
        world2grid = load_matrix(os.path.join(opt.frames, scan_name, 'world2grid.txt'))
        mesh_vertices = util_3d.read_mesh_vertices(mesh_file)
        instances = load_pred(os.path.join(opt.pred_dir, pred_file))
        gt_data = gt_reader(opt.groundtruth_path, '/mnt/raid/ji/mask_rcnn_2dto3d/samples/scannet/filelist/nyu40labels.csv')
        blob = gt_data.get_gt(scan_name + '__0__', '/mnt/raid/ji/mask_rcnn_2dto3d/samples/scannet/filelist/nyu40labels.csv')
        export(mesh_vertices, world2grid, instances, blob, os.path.join(opt.output_dir, scan_name))
        #visualize(mesh_vertices, world2grid, instances, blob)



if __name__ == '__main__':
    main()
