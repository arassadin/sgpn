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
parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 4]')
parser.add_argument('--pretrain', type=bool, default=True, help='pretrain semantics segmenation')
parser.add_argument('--point_num', type=int, default=8192, help='Point Number')
parser.add_argument('--group_num', type=int, default=100, help='Maximum Group Number in one pc')
parser.add_argument('--cate_num', type=int, default=21, help='Number of categories')
parser.add_argument('--margin_same', type=float, default=1., help='Double hinge loss margin: same semantic')
parser.add_argument('--margin_diff', type=float, default=2., help='Double hinge loss margin: different semantic')

# Input&Output Settings
parser.add_argument('--output_dir', type=str, default='checkpoint/scannet_sem_seg2', help='Directory that stores all training logs and trained models')
parser.add_argument('--restore_model', type=str, default='checkpoint/scannet_sem_seg2', help='Pretrained model')

FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

# dataloader
DATA_ROOT = '/mnt/raid/ji/SGPN/data/scannet_data/annotation'
TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_ROOT, npoints=FLAGS.point_num, split='/mnt/raid/ji/SGPN/data/scannet_data/meta/scannet_train.txt')
TEST_DATASET = scannet_dataset.ScannetDatasetWholeScene(root=DATA_ROOT, npoints=FLAGS.point_num, split='/mnt/raid/ji/SGPN/data/scannet_data/meta/scannet_test_few.txt')


PRETRAINED_MODEL_PATH = os.path.join(FLAGS.restore_model, 'trained_models/')
PRETRAIN = FLAGS.pretrain
POINT_NUM = FLAGS.point_num
BATCH_SIZE = FLAGS.batch
OUTPUT_DIR = FLAGS.output_dir

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NUM_GROUPS = FLAGS.group_num
NUM_CATEGORY = FLAGS.cate_num

print('#### Batch Size: {0}'.format(BATCH_SIZE))
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

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, POINT_NUM, 9))
    batch_label = np.zeros((bsize, POINT_NUM), dtype=np.int32)
    batch_group = np.zeros((bsize, POINT_NUM), dtype=np.int32)
    batch_smpw = np.zeros((bsize, POINT_NUM), dtype=np.float32)
    for i in range(bsize):
        ps,seg,group,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_group[i,:] = group
        batch_smpw[i,:] = smpw
        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_group, batch_smpw

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch*BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True
            )
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            batch = tf.Variable(0, trainable=False, name='batch')
            learning_rate = tf.train.exponential_decay(
                BASE_LEARNING_RATE,  # base learning rate
                batch * BATCH_SIZE,  # global_var indicating the number of steps
                DECAY_STEP,  # step size
                DECAY_RATE,  # decay rate
                staircase=True  # Stair-case or continuous decreasing
            )
            bn_decay = get_bn_decay(batch)
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)

            pointclouds_ph, ptsseglabel_ph, ptsseglabel_onehot_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph = \
                model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)
            is_training_ph = tf.placeholder(tf.bool, shape=())

            labels = {'ptsgroup': ptsgroup_label_ph,
                      'semseg': ptsseglabel_ph,
                      'semseg_onehot': ptsseglabel_onehot_ph,
                      'semseg_mask': pts_seglabel_mask_ph,
                      'group_mask': pts_group_mask_ph}

            net_output = model.get_model(pointclouds_ph, is_training_ph, group_cate_num=NUM_CATEGORY, m=MARGINS[0], bn_decay=bn_decay)
            ptsseg_loss, simmat_loss, loss, grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt = model.get_loss(net_output, labels, alpha_ph, MARGINS)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            group_err_loss_ph = tf.placeholder(tf.float32, shape=())
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            group_err_op = tf.summary.scalar('group_err_loss', group_err_loss_ph)

        train_variables = tf.trainable_variables()

        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)
        train_op_pretrain = trainer.minimize(ptsseg_loss, var_list=train_variables, global_step=batch)
        train_op_5epoch = trainer.minimize(simmat_loss, var_list=train_variables, global_step=batch)

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

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            printout(flog, "Fail to load modelfile: %s" % PRETRAINED_MODEL_PATH)


        ## load test data into memory
        test_data = []
        test_group = []
        test_seg = []
        test_smpw = []
        for i in range(len(TEST_DATASET)):
            print(i)
            cur_data, cur_seg, cur_group, cur_smpw = get_test_batch(TEST_DATASET, i)
            test_data += [cur_data]
            test_group += [cur_group]
            test_seg += [cur_seg]
            test_smpw += [cur_smpw]

        test_data = np.concatenate(test_data,axis=0)
        test_group = np.concatenate(test_group,axis=0)
        test_seg = np.concatenate(test_seg,axis=0)
        test_smpw = np.concatenate(test_smpw,axis=0)
        num_data_test = test_data.shape[0]
        num_batch_test = num_data_test // BATCH_SIZE

        def train_one_epoch(epoch_num):

            ### NOTE: is_training = False: We do not update bn parameters during training due to the small batch size. This requires pre-training PointNet with large batchsize (say 32).
            if PRETRAIN:
                is_training = True
            else:
                is_training = False

            total_loss = 0.0
            total_grouperr = 0.0
            total_same = 0.0
            total_diff = 0.0
            total_pos = 0.0
            same_cnt0 = 0

            train_idxs = np.arange(0, len(TRAIN_DATASET))
            np.random.shuffle(train_idxs)
            num_batches = len(TRAIN_DATASET)//BATCH_SIZE
            for batch_idx in range(num_batches):
                print('{}/{}'.format(batch_idx, num_batches))
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                batch_data, batch_label, batch_group, batch_smpw = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
                aug_data = provider.rotate_point_cloud_z(batch_data)
                pts_label_one_hot = model.convert_seg_to_one_hot(batch_label)

                if PRETRAIN:
                    feed_dict = {
                        pointclouds_ph: aug_data, 
                        ptsseglabel_ph: batch_label,
                        ptsseglabel_onehot_ph: pts_label_one_hot,
                        pts_seglabel_mask_ph: batch_smpw,
                        is_training_ph: is_training,
                        alpha_ph: min(10., (float(epoch_num) / 5.) * 2. + 2.),
                    }
                    _, loss_val  = sess.run([train_op_pretrain, ptsseg_loss], feed_dict=feed_dict)
                    total_loss += loss_val
                    if batch_idx % 10 == 9:
                        printout(flog, 'Batch: %d, loss: %f' % (batch_idx, total_loss/10))
                        total_loss = 0.0
                else:
                    pts_group_label, pts_group_mask = model.convert_groupandcate_to_one_hot(batch_group)
                    feed_dict = {
                        pointclouds_ph: batch_data,
                        ptsseglabel_ph: batch_label,
                        ptsseglabel_onehot_ph: pts_label_one_hot,
                        pts_seglabel_mask_ph: batch_smpw,
                        ptsgroup_label_ph: pts_group_label,
                        pts_group_mask_ph: pts_group_mask,
                        is_training_ph: is_training,
                        alpha_ph: min(10., (float(epoch_num) / 5.) * 2. + 2.),
                    }

                    if epoch_num < 20:
                        _, loss_val, simmat_val, grouperr_val, same_val, same_cnt_val, diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run([train_op_5epoch, simmat_loss, net_output['simmat'], grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)
                    else:
                        _, loss_val, simmat_val, grouperr_val, same_val, same_cnt_val, diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run([train_op, loss, net_output['simmat'], grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)

                    total_loss += loss_val
                    total_grouperr += grouperr_val
                    total_diff += (diff_val / diff_cnt_val)
                    if same_cnt_val > 0:
                        total_same += same_val / same_cnt_val
                        same_cnt0 += 1
                    total_pos += pos_val / pos_cnt_val

                    if batch_idx % 10 == 9:
                        printout(flog, 'Batch: %d, loss: %f, grouperr: %f, same: %f, diff: %f, pos: %f' % (batch_idx, total_loss/10, total_grouperr/10, total_same/same_cnt0, total_diff/10, total_pos/10))

                        lr_sum, batch_sum, train_loss_sum, group_err_sum = sess.run( \
                            [lr_op, batch, total_train_loss_sum_op, group_err_op], \
                            feed_dict={total_training_loss_ph: total_loss / 10.,
                                       group_err_loss_ph: total_grouperr / 10., })

                        train_writer.add_summary(train_loss_sum, batch_sum)
                        train_writer.add_summary(lr_sum, batch_sum)
                        train_writer.add_summary(group_err_sum, batch_sum)

                        total_grouperr = 0.0
                        total_loss = 0.0
                        total_diff = 0.0
                        total_same = 0.0
                        total_pos = 0.0
                        same_cnt0 = 0

            cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch_num + 1) + '.ckpt'))
            printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

        def val_one_epoch(epoch_num):
            is_training = False

            def evaluate_confusion(confusion_matrix, epoch):
                conf = confusion_matrix.value()
                total_correct = 0
                valids = np.zeros(NUM_CATEGORY, dtype=np.float32)
                for c in range(NUM_CATEGORY):
                    num = conf[c,:].sum()
                    valids[c] = -1 if num == 0 else float(conf[c][c]) / float(num)
                    total_correct += conf[c][c]
                instance_acc = -1 if conf.sum() == 0 else float(total_correct) / float(conf.sum())
                avg_acc = -1 if np.all(np.equal(valids, -1)) else np.mean(valids[np.not_equal(valids, -1)])
                print('Epoch: {}\tAcc(inst): {:.6f}\tAcc(avg): {:.6f}'.format(epoch, instance_acc, avg_acc))
                for class_ind, class_acc in enumerate(valids[np.not_equal(valids, -1)]):
                    print('{}: {}'.format(class_ind, class_acc))
                with open(os.path.join(LOG_STORAGE_PATH, 'ACC_{}.txt'.format(epoch)), 'w') as f:
                    f.write('Epoch: {}\tAcc(inst): {:.6f}\tAcc(avg): {:.6f}'.format(epoch, instance_acc, avg_acc))
                    for class_ind, class_acc in enumerate(valids[np.not_equal(valids, -1)]):
                        f.write('{}: {}\n'.format(class_ind, class_acc))

            confusion_val = tnt.meter.ConfusionMeter(NUM_CATEGORY)
            for j in range(0, num_batch_test):
                print('{}/{}'.format(j, num_batch_test))
                start_idx = j * BATCH_SIZE
                end_idx = (j + 1) * BATCH_SIZE
                pts_label_one_hot = model.convert_seg_to_one_hot(test_seg[start_idx:end_idx])
                feed_dict = {
                    pointclouds_ph: test_data[start_idx:end_idx,...],
                    ptsseglabel_ph: test_seg[start_idx:end_idx],
                    ptsseglabel_onehot_ph: pts_label_one_hot,
                    pts_seglabel_mask_ph: test_smpw[start_idx:end_idx, ...],
                    is_training_ph: is_training,
                    alpha_ph: min(10., (float(epoch_num) / 5.) * 2. + 2.),
                }

                ptsclassification_val0 = sess.run([net_output['semseg']], feed_dict=feed_dict)
                ptsclassification_val = torch.from_numpy(ptsclassification_val0[0]).view(-1, NUM_CATEGORY)
                ptsclassification_gt = torch.from_numpy(pts_label_one_hot).view(-1, NUM_CATEGORY)
                #import ipdb
                #ipdb.set_trace()
                #pc_util.write_obj_color(np.reshape(test_data[:BATCH_SIZE,:,:3], [-1,3])[:,:3], np.argmax(ptsclassification_val.numpy(), 1), 'pred3.obj')
                #pc_util.write_obj_color(np.reshape(test_data[:BATCH_SIZE,:,:3], [-1,3])[:,:3], np.argmax(ptsclassification_gt.numpy(), 1), 'gt.obj')
                confusion_val.add(target=ptsclassification_gt, predicted=ptsclassification_val)
            evaluate_confusion(confusion_val, epoch_num)


        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
            train_one_epoch(epoch)
            flog.flush()
            if PRETRAIN:
                val_one_epoch(epoch)

        flog.close()


if __name__ == '__main__':
    train()
