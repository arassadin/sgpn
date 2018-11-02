import os
import sys
import numpy as np
import h5py
import csv
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
#
# # Download dataset for point cloud classification
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#     www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#     zipfile = os.path.basename(www)
#     os.system('wget %s; unzip %s' % (www, zipfile))
#     os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#     os.system('rm %s' % (zipfile))

def load_mapping(filename):
    mapping ={0:20}
    csvfile = open(filename) 
    spamreader = csv.DictReader(csvfile, delimiter=',')
    for row in spamreader:
        mapping[int(row['nyu40id'])] = int(row['mappedIdConsecutive'])
    csvfile.close()
    return mapping

mapping = load_mapping('data/scannet_data/meta/nyu40labels_scannet.csv')
def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, :, :3]
        rotated_data[k, :, :3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def save_h5_output(h5_filename, seg, segrefine, group, grouppred, label_dtype='uint8'):
    print(h5_filename)
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'seglabel', data=seg,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'segrefine', data=segrefine,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'pid', data=group,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'predpid', data=grouppred,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def loadDataFile_with_grouplabel(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    # label = f['label'][:]
    group = f['pid'][:]#Nx1
    if 'groupcategory' in f:
        cate = f['groupcategory'][:]#Gx1
    else:
        cate = 0
    return (data, group, cate)

def loadDataFile_with_groupseglabel(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    # label = f['label'][:]
    group = f['pid'][:]#Nx1
    if 'groupcategory' in f:
        cate = f['groupcategory'][:]#Gx1
    else:
        cate = 0
    seg = -1 * np.ones_like(group)
    for i in range(group.shape[0]):
        for j in range(group.shape[1]):
            if group[i,j,0]!=-1 and cate[i,group[i,j,0],0]!=-1:
                    seg[i,j,0] = cate[i,group[i,j,0],0]
    return (data, group, cate, seg)

def loadDataFile_with_groupseglabel_sunrgbd(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    group = f['pid'][:]#NxG
    if 'groupcategory' in f:
        cate = f['groupcategory'][:]#Gx1
    else:
        cate = 0
    if 'seglabel' in f:
        seg = f['seglabel'][:]
    else:
        seg = f['seglabels'][:]
    return (data, group, cate, seg)

def save_3d_points(verts, colors, output_file):
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
        for v, c in zip(verts, colors):
            f.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(v[0], v[1], v[2], ID_COLOR[int(c)%41][0], ID_COLOR[int(c)%41][1], ID_COLOR[int(c)%41][2]))

def loadDataFile_with_groupseglabel_scannet(filename, counter):
    print(counter)
    f = h5py.File(filename)

    data = f['data'][:]
    group = f['pid'][:].astype(np.int32)#NxG

    if 'label' in f:
        label = f['label'][:].astype(np.int32)
    else:
        label = []

    if 'seglabel' in f:
        seg = f['seglabel'][:].astype(np.int32)
    else:
        seg = f['seglabels'][:].astype(np.int32)

    #group = np.reshape(group, [-1])
    #data = np.reshape(data, [-1, 9])
    #save_3d_points(data[:, 6:], group, output_file='test.ply')

    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            seg[i,j] = mapping[seg[i,j]]

    return (data, group, label, seg)


def loadDataFile_with_groupseglabel_nuyv2(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    group = f['pid'][:]#NxG
    if 'groupcategory' in f:
        cate = f['groupcategory'][:]#Gx1
    else:
        cate = 0
    if 'seglabel' in f:
        seg = f['seglabel'][:]
    else:
        seg = f['seglabels'][:]
    boxes = f['bbox'][:]
    return (data, group, cate, seg, boxes)

def loadDataFile_with_groupseglabel_stanfordindoor(filename, counter):
    print(counter)
    f = h5py.File(filename)

    data = f['data'][:]
    group = f['pid'][:].astype(np.int32)#NxG
    if 'label' in f:
        label = f['label'][:].astype(np.int32)
    else :
        label = []
    if 'seglabel' in f:
        seg = f['seglabel'][:].astype(np.int32)
    else:
        seg = f['seglabels'][:].astype(np.int32)

    #for i in range(seg.shape[0]):
    #    for j in range(seg.shape[1]):
    #        seg[i,j] = mapping[seg[i,j]]

    return (data, group, label, seg)

def loadDataFile_with_img(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    group = f['pid'][:]#NxG
    seg = f['seglabel'][:]
    img = f['img'][:].transpose([2,1,0])
    return (data, group, seg, img)
