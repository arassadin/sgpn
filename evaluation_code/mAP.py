import numpy as np
import math


class DetectionMAP:
    def __init__(self, n_class, pr_samples=11, overlap_threshold=0.25, ignore_class=[]):
        """
        Running computation of average precision of n_class in a bounding box + classification task
        :param n_class:             quantity of class
        :param pr_samples:          quantification of threshold for pr curve
        :param overlap_threshold:   minimum overlap threshold
        """
        self.n_class = n_class
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.ignore_class = ignore_class
        self.reset_accumulators()

    def reset_accumulators(self):
        """
        Reset the accumulators state
        TODO this is hard to follow... should use a better data structure
        total_accumulators : list of list of accumulators at each pr_scale for each class
        :return:
        """
        self.total_accumulators = []
        for i in range(self.n_class):
            self.total_accumulators.append(APAccumulator())

    def evaluate_mask(self, pred_mask, pred_classes, pred_box, pred_conf, gt_mask, gt_classes, gt_box):
        """

        :param pred_classes:
        :param gt_classes:
        :param pred_conf:
        :param pred_mask:
        :param gt_mask:
        :return:
        """
        self.evaluate_mask_(self.total_accumulators, pred_mask, pred_classes, pred_box, pred_conf, gt_mask, gt_classes, gt_box)

    def evaluate_mask_(self, accumulators, pred_mask, pred_classes, pred_box, pred_conf, gt_mask, gt_classes, gt_box):

        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        pred_size = pred_classes.shape[0]
        IoU = None
        if pred_size != 0:
            IoU = DetectionMAP.compute_IoU_mask(pred_mask, pred_box, gt_mask, gt_box)
            # mask irrelevant overlaps
            IoU[IoU < self.overlap_threshold] = 0

        # Final match : 1 prediction per GT
        for i, acc in enumerate(accumulators):
            # true positive
            # TP: predict correctly
            # FP: predict wrongly
            # FN: gt missing
            TP, FP, FN = DetectionMAP.compute_TP_FP_FN(pred_classes, gt_classes, pred_conf, IoU, i)
            acc.inc_predictions(TP, FP)
            acc.inc_not_predicted(FN)


    def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
        """
        Update the accumulator for the running mAP evaluation.
        For exemple, this can be called for each images
        :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, z1, x2, y2, z2] :     Shape [n_pred, 6]
        :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
        :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
        :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, z1, x2, y2, z2] :  Shape [n_gt, 6]
        :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
        :return:
        """

        assert pred_bb.ndim == 2
        self.evaluate_(self.total_accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, self.overlap_threshold)

    @staticmethod
    def evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, overlap_threshold=0.5):
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        pred_size = pred_classes.shape[0]
        IoU = None
        if pred_size != 0:
            IoU = DetectionMAP.compute_IoU(pred_bb, gt_bb, pred_conf)
            # mask irrelevant overlaps
            IoU[IoU < overlap_threshold] = 0

        # Final match : 1 prediction per GT
        for i, acc in enumerate(accumulators):
            # true positive
            # TP: predict correctly
            # FP: predict wrongly
            # FN: gt missing
            TP, FP, FN = DetectionMAP.compute_TP_FP_FN(pred_classes, gt_classes, pred_conf, IoU, i)
            acc.inc_predictions(TP, FP)
            acc.inc_not_predicted(FN)


    @staticmethod
    def compute_IoU(prediction, gt, confidence):
        IoU = DetectionMAP.jaccard(prediction, gt)
        return IoU

    @staticmethod
    def compute_IoU_mask(pred_mask, pred_box, gt_mask, gt_box):
        IoU = DetectionMAP.jaccard_mask(pred_mask, pred_box, gt_mask, gt_box)
        return IoU

    @staticmethod
    def intersect_area(box_a, box_b):
        """
        Compute the area of intersection between two rectangular bounding box
        Bounding boxes use corner notation : [x1, y1, z1, x2, y2, z2]
        Args:
          box_a: (np.array) bounding boxes, Shape: [A,6].
          box_b: (np.array) bounding boxes, Shape: [B,6].
        Return:
          np.array intersection area, Shape: [A,B].
        """
        resized_A = box_a[:, np.newaxis, :]
        resized_B = box_b[np.newaxis, :, :]
        max_xyz = np.minimum(resized_A[:, :, 3:], resized_B[:, :, 3:])
        min_xyz = np.maximum(resized_A[:, :, :3], resized_B[:, :, :3])

        diff_xy = (max_xyz - min_xyz)
        inter = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
        return inter[:, :, 0] * inter[:, :, 1] * inter[:,:,2]

    @staticmethod
    def jaccard(box_a, box_b):
        """
        Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (np.array) Predicted bounding boxes,    Shape: [n_pred, 4]
            box_b: (np.array) Ground Truth bounding boxes, Shape: [n_gt, 4]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        inter = DetectionMAP.intersect_area(box_a, box_b)
        area_a = ((box_a[:, 3] - box_a[:, 0]) * (box_a[:, 4] - box_a[:, 1]) * (box_a[:, 5] - box_a[:, 2]))
        area_b = ((box_b[:, 3] - box_b[:, 0]) * (box_b[:, 4] - box_b[:, 1]) * (box_b[:, 5] - box_b[:, 2]))
        area_a = area_a[:, np.newaxis]
        area_b = area_b[np.newaxis, :]
        union = area_a + area_b - inter
        return inter / union

    @staticmethod
    def element_and_or(mask_a, mask_b):
        shape = mask_a.shape
        total_voxels = shape[0]*shape[1]*shape[2]

        mask_count = np.zeros(2, dtype=np.float32)
        mask_a = mask_a.astype(np.float32)
        mask_b = mask_b.astype(np.float32)

        mask_a_gpu = cuda.mem_alloc(mask_a.nbytes)
        mask_b_gpu = cuda.mem_alloc(mask_b.nbytes)
        mask_count_gpu = cuda.mem_alloc(mask_count.nbytes)
        cuda.memcpy_htod(mask_a_gpu, mask_a)
        cuda.memcpy_htod(mask_b_gpu, mask_b)
        cuda.memcpy_htod(mask_count_gpu, mask_count)

        mod = SourceModule("""
                #include <stdio.h>
                __global__ void element_and_or(float* mask_a, float *mask_b, float *mask_count, int dimX, int dimY, int dimZ)
                {
                    int tid = threadIdx.x;
                    __shared__ int sdata_and[512];
                    __shared__ int sdata_or[512];

                    int idx = tid;
                    while (idx < dimX * dimY * dimZ)
                    {
                        if (mask_a[idx] == 1 && mask_b[idx] == 1)
                            sdata_and[tid] += 1;

                        if (mask_a[idx] == 1 || mask_b[idx] == 1)
                            sdata_or[tid] += 1;

                        idx += 512;
                    }

                    __syncthreads();

                    if (tid < 256)
                    {
                        sdata_and[tid] += sdata_and[tid + 256];
                        sdata_or[tid] += sdata_or[tid + 256]; 
                        __syncthreads();  
                    }
                    if (tid < 128) 
                    {
                        sdata_and[tid] += sdata_and[tid + 128];
                        sdata_or[tid] += sdata_or[tid + 128]; 
                        __syncthreads();  
                    }
                    if (tid < 64)  
                    {
                        sdata_and[tid] += sdata_and[tid + 64]; 
                        sdata_or[tid] += sdata_or[tid + 64];  
                        __syncthreads();  
                    }

                    if (tid < 32) 
                    {
                        sdata_and[tid] += sdata_and[tid + 32];
                        sdata_and[tid] += sdata_and[tid + 16];
                        sdata_and[tid] += sdata_and[tid + 8];
                        sdata_and[tid] += sdata_and[tid + 4];
                        sdata_and[tid] += sdata_and[tid + 2];
                        sdata_and[tid] += sdata_and[tid + 1];

                        sdata_or[tid] += sdata_or[tid + 32];
                        sdata_or[tid] += sdata_or[tid + 16];
                        sdata_or[tid] += sdata_or[tid + 8];
                        sdata_or[tid] += sdata_or[tid + 4];
                        sdata_or[tid] += sdata_or[tid + 2];
                        sdata_or[tid] += sdata_or[tid + 1];
                    }

                    if (tid == 0) 
                    {
                        mask_count[0] = sdata_and[0];
                        mask_count[1] = sdata_or[0];
                    }
                }
                """)
        func = mod.get_function('element_and_or')
        func(mask_a_gpu, mask_b_gpu, mask_count_gpu, 
             np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]), 
             block=(512, 1, 1))
        cuda.memcpy_dtoh(mask_count, mask_count_gpu)
        return mask_count[0], mask_count[1]


    @staticmethod
    def jaccard_mask(mask_a, box_a, mask_b, box_b):
        """
        Compute the jaccard overlap of two sets of masks.  The jaccard overlap
        is simply the intersection over union of two masks.  Here we operate on
        ground truth masks and predict mask.
        E.g.:
            A ∩ B / A ∪ B = sum(A.*B > 0) / sum(A.+B > 0)
        Args:
            mask_a: (np.array) Predicted mask,    Shape: [n_mask, mask_width, mask_height, mask_length]
            mask_b: (np.array) Ground Truth mask, Shape: [n_gt, mask_width, mask_height, mask_length]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        iou = np.zeros((len(mask_a), len(mask_b)))
        for i, mask_i in enumerate(mask_a):
            for j, mask_j in enumerate(mask_b):
                both_box = [math.floor(min(box_a[i][0], box_b[j][0])), 
                            math.floor(min(box_a[i][1], box_b[j][1])),
                            math.floor(min(box_a[i][2], box_b[j][2])),
                            math.ceil(max(box_a[i][3], box_b[j][3])),
                            math.ceil(max(box_a[i][4], box_b[j][4])),
                            math.ceil(max(box_a[i][5], box_b[j][5]))]
                mask_i_big = np.zeros((both_box[3] - both_box[0], both_box[4] - both_box[1], both_box[5] - both_box[2]))
                mask_j_big = np.zeros((both_box[3] - both_box[0], both_box[4] - both_box[1], both_box[5] - both_box[2]))
                mask_i_big[int(box_a[i][0]-both_box[0]):int(box_a[i][3]-both_box[0]), 
                           int(box_a[i][1]-both_box[1]):int(box_a[i][4]-both_box[1]), 
                           int(box_a[i][2]-both_box[2]):int(box_a[i][5]-both_box[2])] = mask_i
                mask_j_big[int(box_b[j][0]-both_box[0]):int(box_b[j][3]-both_box[0]), 
                           int(box_b[j][1]-both_box[1]):int(box_b[j][4]-both_box[1]), 
                           int(box_b[j][2]-both_box[2]):int(box_b[j][5]-both_box[2])] = mask_j
                intersect = sum(sum(sum(mask_i_big * mask_j_big > 0)))
                overall = sum(sum(sum(mask_i_big + mask_j_big > 0)))
                #intersect, overall = DetectionMAP.element_and_or(mask_i_big, mask_j_big)
                iou[i, j] = intersect / overall

        return iou

    @staticmethod
    def compute_TP_FP_FN(pred_cls, gt_cls, pred_conf, IoU, class_index):
        if pred_cls.shape[0] == 0:
            return [], [], sum(gt_cls == class_index)

        IoU_mask = IoU != 0

        IoU_mask = IoU_mask[pred_cls == class_index, :]
        IoU_mask = IoU_mask[:, gt_cls == class_index]

        # IoU number for multiple gt on one pred
        IoU = IoU[pred_cls == class_index,:]
        IoU = IoU[:, gt_cls == class_index]

        # sum all gt with prediction of this class
        TP = []
        FP = []
        FN = sum(gt_cls == class_index)
        sort_conf_arg = np.argsort(pred_conf[pred_cls == class_index])[::-1]
        for i in sort_conf_arg:
            ind = -1
            max_overlapping = -1
            for j in range(IoU_mask.shape[1]):
                if IoU_mask[i,j] == True and IoU[i,j] > max_overlapping:
                    max_overlapping = IoU[i, j]
                    ind = j
            if ind != -1:
                TP.append(pred_conf[i])
                IoU_mask[:, ind] = False
                FN -= 1
            else:
                FP.append(pred_conf[i])
        return TP, FP, FN


    def compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        previous_precision = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += previous_precision * (recall - previous_recall)
            previous_recall = recall
            previous_precision = precision
        return average_precision

    def compute_precision_recall_(self, class_index, interpolated=True):
        precisions = []
        recalls = []
        acc = self.total_accumulators[class_index]
        for i in self.pr_scale:
            precision, recall  = acc.precision_recall(i)
            if recall == 1.0 and precision == 0.0:
                break
            else:
                recalls.append(recall)
                precisions.append(precision)

        precisions = precisions[::-1]
        recalls = recalls[::-1]
        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions, recalls

    def plot_pr(self, ax, class_index, precisions, recalls, average_precision):
        ax.step(recalls, precisions, color='b', alpha=0.2,
                where='post')
        ax.fill_between(recalls, precisions, step='post', alpha=0.2,
                        color='b')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('cls {0:} : AUC={1:0.2f}'.format(class_index, average_precision))

    def plot(self, interpolated=False):
        """
        Plot all pr-curves for each classes
        :param interpolated: will compute the interpolated curve
        :return:
        """
        grid = int(math.ceil(math.sqrt(self.n_class)))
        fig, axes = plt.subplots(nrows=grid, ncols=grid)
        mean_average_precision = []
        # TODO: data structure not optimal for this operation...
        axes = np.array(axes)
        for i, ax in enumerate(axes.flat):
            if i > self.n_class - 1:
                break
            
            precisions, recalls = self.compute_precision_recall_(i, interpolated)
            average_precision = self.compute_ap(precisions, recalls)
            self.plot_pr(ax, i, precisions, recalls, average_precision)
            mean_average_precision.append(average_precision)

        plt.suptitle("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))

    def mAP(self):
        mean_average_precision = []
        for i in range(self.n_class):
            if i in self.ignore_class:
                continue
            precisions, recalls = self.compute_precision_recall_(i, True)
            average_precision = self.compute_ap(precisions, recalls)
            mean_average_precision.append(average_precision)

        if len(mean_average_precision) == 0:
            return 0
        else:
            return sum(mean_average_precision)/len(mean_average_precision)

    def AP(self, idx):
        """

        :param idx:
        :return:
        """
        precisions, recalls = self.compute_precision_recall_(idx, True)
        average_precision = self.compute_ap(precisions, recalls)
        return average_precision

    def finalize(self):
        for acc_ind, acc in enumerate(self.total_accumulators):
            acc.ranking()
            if acc.if_ignore() and acc_ind not in self.ignore_class:
                self.ignore_class.append(acc_ind)


class DetectionMAP_2D:
    def __init__(self, n_class, pr_samples=11, overlap_threshold=0.25, ignore_class=[]):
        """
        Running computation of average precision of n_class in a bounding box + classification task
        :param n_class:             quantity of class
        :param pr_samples:          quantification of threshold for pr curve
        :param overlap_threshold:   minimum overlap threshold
        """
        self.n_class = n_class
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.ignore_class = ignore_class
        self.reset_accumulators()

    def reset_accumulators(self):
        """
        Reset the accumulators state
        TODO this is hard to follow... should use a better data structure
        total_accumulators : list of list of accumulators at each pr_scale for each class
        :return:
        """
        self.total_accumulators = []
        for i in range(self.n_class):
            self.total_accumulators.append(APAccumulator())

    def evaluate_mask(self, pred_mask, pred_classes, pred_box, pred_conf, gt_mask, gt_classes, gt_box):
        """

        :param pred_classes:
        :param gt_classes:
        :param pred_conf:
        :param pred_mask:
        :param gt_mask:
        :return:
        """
        self.evaluate_mask_(self.total_accumulators, pred_mask, pred_classes, pred_box, pred_conf, gt_mask, gt_classes, gt_box)

    def evaluate_mask_(self, accumulators, pred_mask, pred_classes, pred_box, pred_conf, gt_mask, gt_classes, gt_box):

        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        pred_size = pred_classes.shape[0]
        IoU = None
        if pred_size != 0:
            IoU = DetectionMAP_2D.compute_IoU_mask(pred_mask, pred_box, gt_mask, gt_box, pred_conf)
            # mask irrelevant overlaps
            IoU[IoU < self.overlap_threshold] = 0

        # Final match : 1 prediction per GT
        for i, acc in enumerate(accumulators):
            # true positive
            # TP: predict correctly
            # FP: predict wrongly
            # FN: gt missing
            TP, FP, FN = DetectionMAP_2D.compute_TP_FP_FN(pred_classes, gt_classes, pred_conf, IoU, i)
            acc.inc_predictions(TP, FP)
            acc.inc_not_predicted(FN)


    def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
        """
        Update the accumulator for the running mAP evaluation.
        For exemple, this can be called for each images
        :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, z1, x2, y2, z2] :     Shape [n_pred, 6]
        :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
        :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
        :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, z1, x2, y2, z2] :  Shape [n_gt, 6]
        :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
        :return:
        """

        assert pred_bb.ndim == 2
        self.evaluate_(self.total_accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, self.overlap_threshold)

    @staticmethod
    def evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, overlap_threshold=0.5):
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        pred_size = pred_classes.shape[0]
        IoU = None
        if pred_size != 0:
            IoU = DetectionMAP_2D.compute_IoU(pred_bb, gt_bb, pred_conf)
            # mask irrelevant overlaps
            IoU[IoU < overlap_threshold] = 0

        # Final match : 1 prediction per GT
        for i, acc in enumerate(accumulators):
            # true positive
            # TP: predict correctly
            # FP: predict wrongly
            # FN: gt missing
            TP, FP, FN = DetectionMAP_2D.compute_TP_FP_FN(pred_classes, gt_classes, pred_conf, IoU, i)
            acc.inc_predictions(TP, FP)
            acc.inc_not_predicted(FN)


    @staticmethod
    def compute_IoU(prediction, gt, confidence):
        IoU = DetectionMAP_2D.jaccard(prediction, gt)
        return IoU

    @staticmethod
    def compute_IoU_mask(prediction, pred_box, gt, gt_box, confidence):
        IoU = DetectionMAP_2D.jaccard_mask(prediction, pred_box, gt, gt_box)
        return IoU

    @staticmethod
    def intersect_area(box_a, box_b):
        """
        Compute the area of intersection between two rectangular bounding box
        Bounding boxes use corner notation : [x1, y1, z1, x2, y2, z2]
        Args:
          box_a: (np.array) bounding boxes, Shape: [A,6].
          box_b: (np.array) bounding boxes, Shape: [B,6].
        Return:
          np.array intersection area, Shape: [A,B].
        """
        resized_A = box_a[:, np.newaxis, :]
        resized_B = box_b[np.newaxis, :, :]
        max_xy = np.minimum(resized_A[:, :, 2:], resized_B[:, :, 2:])
        min_xy = np.maximum(resized_A[:, :, :2], resized_B[:, :, :2])

        diff_xy = (max_xy - min_xy)
        inter = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
        return inter[:, :, 0] * inter[:, :, 1]

    @staticmethod
    def jaccard(box_a, box_b):
        """
        Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (np.array) Predicted bounding boxes,    Shape: [n_pred, 4]
            box_b: (np.array) Ground Truth bounding boxes, Shape: [n_gt, 4]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        inter = DetectionMAP_2D.intersect_area(box_a, box_b)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        area_a = area_a[:, np.newaxis]
        area_b = area_b[np.newaxis, :]
        union = area_a + area_b - inter
        return inter / union

    @staticmethod
    def jaccard_mask(mask_a, box_a, mask_b, box_b):
        """
        Compute the jaccard overlap of two sets of masks.  The jaccard overlap
        is simply the intersection over union of two masks.  Here we operate on
        ground truth masks and predict mask.
        E.g.:
            A ∩ B / A ∪ B = sum(A.*B > 0) / sum(A.+B > 0)
        Args:
            mask_a: (np.array) Predicted mask,    Shape: [n_mask, mask_width, mask_height, mask_length]
            mask_b: (np.array) Ground Truth mask, Shape: [n_gt, mask_width, mask_height, mask_length]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        inter = DetectionMAP_2D.intersect_area_mask(mask_a, box_a, mask_b, box_b)

        iou = np.zeros((mask_a.shape[0], mask_b.shape[0]))
        for i in range(mask_a.shape[0]):
            for j in range(mask_b.shape[0]):
                if inter[i, j] == 0:
                    iou[i,j] = 0
                else:
                    both_box = [math.floor(min(box_a[i][0], box_b[j][0])), 
                                math.floor(min(box_a[i][1], box_b[j][1])),
                                math.ceil(max(box_a[i][2], box_b[j][2])),
                                math.ceil(max(box_a[i][3], box_b[j][3]))]
                    iou[i,j] = inter[i,j] / sum(sum(mask_a[i][both_box[0]:both_box[2], both_box[1]:both_box[3]] + 
                                                    mask_b[j][both_box[0]:both_box[2], both_box[1]:both_box[3]]> 0))
        return iou

    @staticmethod
    def intersect_area_mask(mask_a, box_a, mask_b, box_b):
        """
        Compute the area of intersection between two pooled size masks
        Args:
           mask_a: (np.array) Predicted mask,    Shape: [n_mask, mask_width, mask_height, mask_length]
           mask_b: (np.array) Ground Truth mask, Shape: [n_gt, mask_width, mask_height, mask_length]
        Return:
          np.array intersection area, Shape: [A,B].
        """
        intersect = np.zeros((mask_a.shape[0], mask_b.shape[0]))
        for i in range(mask_a.shape[0]):
            for j in range(mask_b.shape[0]):
                both_box = [math.floor(min(box_a[i][0], box_b[j][0])), 
                            math.floor(min(box_a[i][1], box_b[j][1])),
                            math.ceil(max(box_a[i][2], box_b[j][2])),
                            math.ceil(max(box_a[i][3], box_b[j][3]))]
                try:
                    intersect[i,j] = sum(sum(mask_a[i][both_box[0]:both_box[2], both_box[1]:both_box[3]] *
                                             mask_b[j][both_box[0]:both_box[2], both_box[1]:both_box[3]]> 0))
                except:
                    intersect[i,j] = 0
        return intersect

    @staticmethod
    def compute_TP_FP_FN(pred_cls, gt_cls, pred_conf, IoU, class_index):
        if pred_cls.shape[0] == 0:
            return [], [], sum(gt_cls == class_index)

        IoU_mask = IoU != 0

        IoU_mask = IoU_mask[pred_cls == class_index, :]
        IoU_mask = IoU_mask[:, gt_cls == class_index]

        # IoU number for multiple gt on one pred
        IoU = IoU[pred_cls == class_index,:]
        IoU = IoU[:, gt_cls == class_index]

        # sum all gt with prediction of this class
        TP = []
        FP = []
        FN = sum(gt_cls == class_index)
        sort_conf_arg = np.argsort(pred_conf[pred_cls == class_index])[::-1]
        for i in sort_conf_arg:
            ind = -1
            max_overlapping = -1
            for j in range(IoU_mask.shape[1]):
                if IoU_mask[i,j] == True and IoU[i,j] > max_overlapping:
                    max_overlapping = IoU[i, j]
                    ind = j
            if ind != -1:
                TP.append(pred_conf[i])
                IoU_mask[:, ind] = False
                FN -= 1
            else:
                FP.append(pred_conf[i])
        return TP, FP, FN


    def compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def compute_precision_recall_(self, class_index, interpolated=True):
        precisions = []
        recalls = []
        acc = self.total_accumulators[class_index]
        for i in self.pr_scale:
            precision, recall = acc.precision_recall(i)
            if precision == None and recall == None:
                break
            else:
                precisions.append(precision)
                recalls.append(recall)

        precisions = precisions[::-1]
        recalls = recalls[::-1]
        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions, recalls

    def plot_pr(self, ax, class_index, precisions, recalls, average_precision):
        ax.step(recalls, precisions, color='b', alpha=0.2,
                where='post')
        ax.fill_between(recalls, precisions, step='post', alpha=0.2,
                        color='b')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('cls {0:} : AUC={1:0.2f}'.format(class_index, average_precision))

    def plot(self, interpolated=False):
        """
        Plot all pr-curves for each classes
        :param interpolated: will compute the interpolated curve
        :return:
        """
        grid = int(math.ceil(math.sqrt(self.n_class)))
        fig, axes = plt.subplots(nrows=grid, ncols=grid)
        mean_average_precision = []
        # TODO: data structure not optimal for this operation...
        axes = np.array(axes)
        for i, ax in enumerate(axes.flat):
            if i > self.n_class - 1:
                break
            precisions, recalls = self.compute_precision_recall_(i, interpolated)
            average_precision = self.compute_ap(precisions, recalls)
            self.plot_pr(ax, i, precisions, recalls, average_precision)
            mean_average_precision.append(average_precision)

        plt.suptitle("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))

    def mAP(self):
        mean_average_precision = []
        for i in range(self.n_class):
            if i in self.ignore_class:
                continue
            precisions, recalls = self.compute_precision_recall_(i, True)
            average_precision = self.compute_ap(precisions, recalls)
            mean_average_precision.append(average_precision)

        if len(mean_average_precision) == 0:
            return 0
        else:
            return sum(mean_average_precision)/len(mean_average_precision)

    def AP(self, idx):
        """

        :param idx:
        :return:
        """
        precisions, recalls = self.compute_precision_recall_(idx, True)
        average_precision = self.compute_ap(precisions, recalls)
        return average_precision

    def finalize(self):
        for acc_ind, acc in enumerate(self.total_accumulators):
            acc.ranking()
            if acc.if_ignore() and acc_ind not in self.ignore_class:
                self.ignore_class.append(acc_ind)


class APAccumulator:

    """
        Simple accumulator class that keeps track of True positive, False positive and False negative
        to compute precision and recall of a certain class

        predition can only be true positive and false postive (both should have conf)
    """

    def __init__(self):
        # tp: 1, fp: 0
        self.predictions = [] 
        self.FN = 0
        self.TP = 0

    def inc_predictions(self, TP, FP):
        for tp in TP:
            self.predictions.append([tp, 1.0])
            self.TP += 1
        for fp in FP:
            self.predictions.append([fp, 0.0])

    def inc_not_predicted(self, value=1):
        self.FN += value

    def ranking(self):
        if len(self.predictions) != 0:
            self.predictions = np.stack(self.predictions, 0)
            argsort = np.argsort(self.predictions[:,0])[::-1]
            self.predictions = self.predictions[argsort]
        else:
            self.predictions = np.empty(shape=(0,0))

    def if_ignore(self):
        total_gt = self.TP + self.FN
        if total_gt == 0:
            return True
        else:
            return False

    def precision_recall(self, thresh):

        if thresh == 0.0:
            return (1.0, 0.0)

        TP = 0.0
        FP = 0.0
        total_gt = self.TP + self.FN

        for i in range(self.predictions.shape[0]):
            if self.predictions[i][1] == 1.0:
                TP += 1
            else:
                FP += 1

            recall = TP / float(total_gt)
            precision = TP / (TP + FP)
            # if reach recall, return
            if recall >= thresh:
                return precision, recall

        return (0.0, 1.0)

    def __str__(self):
        str = ""
        str += "True positives : {}\n".format(sum(self.TP))
        str += "False positives : {}\n".format(sum(self.FP))
        str += "False Negatives : {}\n".format(sum(self.FN))
        str += "Precision : {}\n".format(sum(self.TP)/(sum(self.FP) + sum(self.TP)))
        str += "Recall : {}\n".format(sum(self.TP)/(sum(self.FP) + sum(self.TN)))
        return str

