import numpy as np
import h5py
import csv
from provider import save_3d_points
with open('data/scannet_filelist/train_hdf5_file_list.txt') as f:
    data_lists = f.readlines()
 
def load_mapping(filename):
    mapping ={0:-1}
    csvfile = open(filename) 
    spamreader = csv.DictReader(csvfile, delimiter=',')
    for row in spamreader:
        mapping[int(row['nyu40id'])] = int(row['mappedIdConsecutive'])
    csvfile.close()
    return mapping

mapping = load_mapping('data/nyu40labels_scannet.csv')
class_num = 20
labelweights = np.zeros(class_num)

for f in data_lists:
    print(f.strip())
    fh = h5py.File(f.strip(), 'r')
    seg = fh['seglabel'][:]
#    data = fh['data'][:]
#    data = np.reshape(data, [-1, 9])
#    seg = np.reshape(seg, [-1])

#    save_3d_points(data[:, 6:], seg, output_file='test.ply')

    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            seg[i,j] = mapping[seg[i,j]]
    tmp, _ =np.histogram(seg, range(class_num+1))
    labelweights += tmp
labelweights = labelweights.astype(np.float32)
labelweights = labelweights / np.sum(labelweights)
labelweights = 1 / np.log(1.2 + labelweights)
print(labelweights)

