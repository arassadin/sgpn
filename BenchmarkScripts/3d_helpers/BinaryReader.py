import struct
import csv
import os
import numpy as np
import math

class BinaryReaderEOFException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'Not enough bytes in file to satisfy read request'

class BinaryReader(object):

    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.typeNames = {
            'int8': 'b',
            'uint8': 'B',
            'int16': 'h',
            'uint16': 'H',
            'int32': 'i',
            'uint32': 'I',
            'int64': 'q',
            'uint64': 'Q',
            'float': 'f',
            'double': 'd',
            'char': 's'}

    def read(self, typeName, times=1):
        typeFormat = self.typeNames[typeName.lower()]*times
        typeSize = struct.calcsize(typeFormat)
        value = self.file.read(typeSize)
        if typeSize != len(value):
            raise BinaryReaderEOFException
        return struct.unpack(typeFormat, value)

    def close(self):
        self.file.close()

class gt_reader(object):
    def __init__(self, base_path, labelset=None):
        self.base_path = base_path
        if labelset != None:
            self.mapping = self.load_mapping(labelset)
        self.maxHeight = 48

    def load_mapping(self, label_file):
        mapping = dict()
        # first weight for background 
        csvfile = open(label_file) 
        spamreader = csv.DictReader(csvfile, delimiter=',')
        for row in spamreader:
            mapping[int(row['nyu40id'])] = int(row['mappedIdConsecutive'])
        csvfile.close()
        return mapping

    def get_gt(self, name, labelset=None):
        # read out data
        reader = BinaryReader(os.path.join(self.base_path, name + '.scene'))
        data_dimX, data_dimY, data_dimZ = reader.read('UINT64', 3)
        data = reader.read('float', data_dimX * data_dimY * data_dimZ)
        data = np.expand_dims(np.reshape(data, (data_dimX, data_dimY, data_dimZ), order='F'), 0).astype(np.float32)
        abs_data = np.clip(np.abs(data), -3, 3)

        (num_box,) = reader.read('uint32')
        gt_box = []
        gt_box_ids = []

        for i in range(num_box):
            # gt_box
            minx, miny, minz, maxx, maxy, maxz = reader.read('float', 6)
            (labelid,) = reader.read('uint32')
            if maxy <= self.maxHeight and miny <= self.maxHeight:
                if labelset != None:
                    labelid = self.mapping[labelid]
                gt_box.append([math.floor(minx), math.floor(miny), math.floor(minz), math.ceil(maxx), math.ceil(maxy), math.ceil(maxz), labelid])
                gt_box_ids.append(i)
        gt_box = np.array(gt_box)

        gt_mask = []
        (num_mask,) = reader.read('uint32')

        for i in range(num_mask):
            (labelid,) = reader.read('uint32')
            dimX, dimY, dimZ = reader.read('UINT64', 3)
            mask_data = reader.read('uint16', dimX * dimY * dimZ)
            if i in gt_box_ids:
                mask_data = np.reshape(mask_data, (dimX, dimY, dimZ), order='F').astype(np.uint8)
                mask_data[mask_data > 1] = 0
                gt_mask.append(mask_data)

        # dict return
        dict_return = {
            'id': name,
            'data': abs_data[:,:,:self.maxHeight,:],
            'gt_box': gt_box,
            'gt_mask': gt_mask,
            'dim': (data_dimX, data_dimY, data_dimZ)
        }
        reader.close()

        return dict_return


