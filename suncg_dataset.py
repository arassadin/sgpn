import pickle
import os
import sys
import numpy as np
import csv
import h5py
from utils import pc_util
from utils import scene_util

NUM_CLASSES = 28

class SUNCGDataset():
    def __init__(self, root, npoints=4096, split='train'):
        self.npoints = npoints
        with open(split) as f:
            scenes = f.readlines()
        self.data_filenames = [scene.strip() for scene in scenes]
        self.scene_points_list = [] 
        self.semantic_labels_list = []
        self.instance_labels_list = []
        #self.mapping = self.load_mapping('data/scannet_data/meta/nyu40labels_scannet.csv')
        self.mapping = self.load_mapping('data/suncg_data/meta/nyu40labels_suncg.csv')

        for idx, data_frame in enumerate(self.data_filenames):
            print(idx)
            data = np.load(os.path.join(root, data_frame + '.npy'))
            for i in range(data.shape[0]):
                data[i, 6] = self.mapping[int(data[i,6])]

            self.scene_points_list += [data[:, :6]]
            self.semantic_labels_list += [data[:, 6]]
            self.instance_labels_list += [data[:, 7]]

        labelweights = np.zeros(NUM_CLASSES)
        for seg in self.semantic_labels_list:
            tmp,_ = np.histogram(seg,range(NUM_CLASSES+1))
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights/np.sum(labelweights)
        self.labelweights = 1/np.log(1.2+labelweights)

    def load_mapping(self, filename):
        mapping ={0:NUM_CLASSES-1}
        csvfile = open(filename) 
        spamreader = csv.DictReader(csvfile, delimiter=',')
        for row in spamreader:
            mapping[int(row['nyu40id'])] = int(row['mappedIdConsecutive'])
        csvfile.close()
        return mapping

    def __getitem__(self, index):
        point_set = self.scene_points_list[index][:,:3]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        instance_seg = self.instance_labels_list[index].astype(np.int32)

        coordmax = np.max(point_set,axis=0)
        coordmin = np.min(point_set,axis=0)

        point_set_normalize = (self.scene_points_list[index][:,:3] - coordmin) / (coordmax - coordmin)
        point_set_rgb = self.scene_points_list[index][:,3:6] / 255.0

        smpmin = np.maximum(coordmax-[1.5,3.0,1.5], coordmin)
        smpmin[1] = coordmin[1]
        smpsz = np.minimum(coordmax-smpmin,[1.5,3.0,1.5])
        smpsz[1] = coordmax[1]-coordmin[1]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter-[0.75, 1.5, 0.75]
            curmax = curcenter+[0.75, 1.5, 0.75]
            curmin[1] = coordmin[1]
            curmax[1] = coordmax[1]
            curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3

            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            cur_instance_seg = instance_seg[curchoice]
            cur_rgb_set = point_set_rgb[curchoice, :]
            cur_normalize_set = point_set_normalize[curchoice, :]

            if len(cur_semantic_seg)==0:
                continue
            mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,62.0,31.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*31.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/62.0/31.0>=0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        instance_seg = cur_instance_seg[choice] 

        rgb_set = cur_rgb_set[choice, :]
        normalize_set = cur_normalize_set[choice,:]
        point_set = np.concatenate([point_set, rgb_set, normalize_set], axis=1)

        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        #import ipdb
        #ipdb.set_trace()
        #pc_util.write_obj_color(point_set[:,:3], semantic_seg, 'semantics.obj')
        #pc_util.write_obj_color(point_set[:,:3], instance_seg, 'instance.obj')


        return point_set, semantic_seg, instance_seg, sample_weight

    def __len__(self):
        return len(self.scene_points_list)

class SUNCGDatasetWholeScene():
    def __init__(self, root, npoints=4096, split='train'):
        self.npoints = npoints
        with open(split) as f:
            scenes = f.readlines()
        self.data_filenames = [scene.strip() for scene in scenes]
        self.scene_points_list = [] 
        self.semantic_labels_list = []
        self.instance_labels_list = []
        #self.mapping = self.load_mapping('data/scannet_data/meta/nyu40labels_scannet.csv')
        self.mapping = self.load_mapping('data/suncg_data/meta/nyu40labels_suncg.csv')

        for data_frame in self.data_filenames:
            data = np.load(os.path.join(root, data_frame + '.npy'))
            for i in range(data.shape[0]):
                data[i, 6] = self.mapping[int(data[i,6])]

            self.scene_points_list += [data[:, :6]]
            self.semantic_labels_list += [data[:, 6]]
            self.instance_labels_list += [data[:, 7]]

        labelweights = np.zeros(NUM_CLASSES)
        for seg in self.semantic_labels_list:
            tmp,_ = np.histogram(seg,range(NUM_CLASSES+1))
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights/np.sum(labelweights)
        self.labelweights = 1/np.log(1.2+labelweights)

    def load_mapping(self, filename):
        mapping ={0:NUM_CLASSES-1}
        csvfile = open(filename) 
        spamreader = csv.DictReader(csvfile, delimiter=',')
        for row in spamreader:
            mapping[int(row['nyu40id'])] = int(row['mappedIdConsecutive'])
        csvfile.close()
        return mapping
    def get_filename(self, index):
        return self.data_filenames[index]

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index][:,:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        instance_seg_ini = self.instance_labels_list[index].astype(np.int32)

        coordmax = np.max(point_set_ini[:,:3],axis=0)
        coordmin = np.min(point_set_ini[:,:3],axis=0)

        point_set_normalize = (self.scene_points_list[index][:,:3] - coordmin) / (coordmax - coordmin)
        point_set_rgb = self.scene_points_list[index][:,3:6] / 255.0

        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        instance_segs = list()
        rgb_sets = list()
        normalize_sets = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*1.5,j*1.5,0]
                curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
                cur_point_set = point_set_ini[curchoice,:]
                cur_rgb_set = point_set_rgb[curchoice, :]
                cur_normalize_set = point_set_normalize[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                cur_instance_seg = instance_seg_ini[curchoice]

                if len(cur_semantic_seg)==0:
                    continue
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)

                mask = np.sum((cur_point_set>=(curmin-0.001))*(cur_point_set<=(curmax+0.001)),axis=1)==3
                mask = mask[choice]
                if sum(mask)/float(len(mask))<0.01:
                    continue

                point_set = cur_point_set[choice,:] # Nx3
                rgb_set = cur_rgb_set[choice, :]
                normalize_set = cur_normalize_set[choice, :]
                semantic_seg = cur_semantic_seg[choice] # N
                instance_seg = cur_instance_seg[choice]

                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN

                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                instance_segs.append(np.expand_dims(instance_seg,0)) # 1xN
                rgb_sets.append(np.expand_dims(rgb_set,0)) # 1xN
                normalize_sets.append(np.expand_dims(normalize_set,0)) # 1xN

        point_sets = np.concatenate(tuple(point_sets),axis=0)
        rgb_sets = np.concatenate(tuple(rgb_sets),axis=0)
        normalize_sets = np.concatenate(tuple(normalize_sets), axis=0)
        point_sets = np.concatenate([point_sets, rgb_sets, normalize_sets], axis=2)

        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        instance_segs = np.concatenate(tuple(instance_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)

        return point_sets, semantic_segs, instance_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)

