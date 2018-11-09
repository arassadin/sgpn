import argparse
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import sys
import torch

from utils import pc_util
from models import model
import torchnet as tnt
import scannet_dataset
import suncg_dataset
from utils import provider
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# need to be changed
# 0. input/output folder 
# 1. filelist
# 2. batch size
# 3. is_training
# 4. '''''' part
# 5. loss

#----------------------------------------------------------
# set 
# Parsing Arguments
parser = argparse.ArgumentParser()
# Experiment Settings
parser.add_argument('--gpu', type=str, default="3", help='GPU to use [default: GPU 1]')
parser.add_argument('--wd', type=float, default=0.9, help='Weight Decay [Default: 0.0]')
parser.add_argument('--epoch', type=int, default=200, help='Number of epochs [default: 50]')
parser.add_argument('--pretrain', type=bool, default=True, help='pretrain semantics segmenation')
parser.add_argument('--point_num', type=int, default=4096, help='Point Number')
parser.add_argument('--group_num', type=int, default=150, help='Maximum Group Number in one pc')
parser.add_argument('--cate_num', type=int, default=21, help='Number of categories')
parser.add_argument('--margin_same', type=float, default=1., help='Double hinge loss margin: same semantic')
parser.add_argument('--margin_diff', type=float, default=2., help='Double hinge loss margin: different semantic')

# Input&Output Settings
parser.add_argument('--output_dir', type=str, default='checkpoint/scannet_sem_seg2', help='Directory that stores all training logs and trained models')
parser.add_argument('--restore_model', type=str, default='checkpoint/scannet_sem_seg2', help='Pretrained model')

FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

# dataloader
#DATA_ROOT = 'data/scannet_data/annotation'
#TEST_DATASET = scannet_dataset.ScannetDatasetWholeScene(root=DATA_ROOT, npoints=FLAGS.point_num, split='data/scannet_data/meta/scannet_test.txt')
DATA_ROOT = 'data/suncg_data/annotation'
TEST_DATASET = suncg_dataset.SUNCGDatasetWholeScene(root=DATA_ROOT, npoints=FLAGS.point_num, split='data/suncg_data/meta/suncg_test.txt')


PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_model, 'trained_models/')
PRETRAIN = FLAGS.pretrain
POINT_NUM = FLAGS.point_num
OUTPUT_DIR = FLAGS.output_dir

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NUM_GROUPS = FLAGS.group_num
NUM_CATEGORY = FLAGS.cate_num

print('#### Point Number: {0}'.format(POINT_NUM))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

DECAY_STEP = 200000.
DECAY_RATE = 0.7

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LEARNING_RATE_CLIP = 1e-5
BASE_LEARNING_RATE = 5e-4

TRAINING_EPOCHES = FLAGS.epoch
MARGINS = [FLAGS.margin_same, FLAGS.margin_diff]

print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(OUTPUT_DIR, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

LOG_DIR = FLAGS.output_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

os.system('cp %s %s' % (os.path.join(BASE_DIR, 'models/model.py'), LOG_DIR))  # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))  # bkp of train procedure

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def get_test_batch(dataset, i):
    batch_data = []
    batch_label = []
    batch_group = []
    batch_smpw = []
    ps,seg,group,smpw = dataset[i]
    batch_data.append(ps)
    batch_label.append(seg)
    batch_group.append(group)
    batch_smpw.append(smpw)
    batch_data = np.concatenate(batch_data, 0)
    batch_label = np.concatenate(batch_label, 0)
    batch_group = np.concatenate(batch_group, 0)
    batch_smpw = np.concatenate(batch_smpw, 0)
    return batch_data, batch_label, batch_group, batch_smpw



def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            batch = tf.Variable(0, trainable=False, name='batch')
            pointclouds_ph, ptsseglabel_ph, ptsseglabel_onehot_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph = \
                model.placeholder_inputs(1, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)
            is_training_ph = tf.placeholder(tf.bool, shape=())

            labels = {'ptsgroup': ptsgroup_label_ph,
                      'semseg': ptsseglabel_ph,
                      'semseg_onehot': ptsseglabel_onehot_ph,
                      'semseg_mask': pts_seglabel_mask_ph,
                      'group_mask': pts_group_mask_ph}

            net_output = model.get_model(pointclouds_ph, is_training_ph, group_cate_num=NUM_CATEGORY, m=MARGINS[0])
            ptsseg_loss, simmat_loss, loss, grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt = model.get_loss(net_output, labels, alpha_ph, MARGINS)

        loader = tf.train.Saver([v for v in tf.all_variables()#])
                                 if
                                   ('conf_logits' not in v.name) and
                                    ('Fsim' not in v.name) and
                                    ('Fsconf' not in v.name) and
                                    ('batch' not in v.name)
                                ])
        saver = tf.train.Saver([v for v in tf.all_variables()], max_to_keep=200)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)

        ## load test data into memory
        is_training = False
        for i in range(len(TEST_DATASET)):
            print('{}/{}'.format(i, len(TEST_DATASET)))
            cur_data, cur_seg, cur_group, cur_smpw = get_test_batch(TEST_DATASET, i)
            data_res = []
            seg_res = []
            for j in range(cur_data.shape[0]):
                pts_label_one_hot = model.convert_seg_to_one_hot(cur_seg[j:j+1])
                feed_dict = {
                    pointclouds_ph: cur_data[j:j+1],
                    ptsseglabel_ph: cur_seg[j:j+1],
                    ptsseglabel_onehot_ph: pts_label_one_hot,
                    pts_seglabel_mask_ph: cur_smpw[j:j+1],
                    is_training_ph: is_training,
                    alpha_ph: min(10., (float(1.0) / 5.) * 2. + 2.),
                }
                ptsclassification_val0 = sess.run([net_output['semseg']], feed_dict=feed_dict)
                ptsclassification_val = torch.from_numpy(ptsclassification_val0[0]).view(-1, NUM_CATEGORY)
                ptsclassification_gt = torch.from_numpy(pts_label_one_hot).view(-1, NUM_CATEGORY)
                data_res.append(cur_data[j:j+1])
                seg_res.append(np.argmax(ptsclassification_val.numpy(), 1))

            write_data = np.reshape(np.concatenate(data_res, 0)[:,:,:3], [-1,3])
            write_seg = np.concatenate(seg_res, 0)
            pc_util.write_ply_res(write_data, write_seg, '{}.ply'.format(TEST_DATASET.get_filename(i)))
            pc_util.write_obj_color(write_data, write_seg, '{}.obj'.format(TEST_DATASET.get_filename(i)))


if __name__ == '__main__':
    eval()
